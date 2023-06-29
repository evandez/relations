import itertools
import logging
import random
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from src import data, functional, models
from src.utils.typing import Layer

import baukit
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RelationOutput:
    """Predicted object tokens and their probabilities under the decoder head."""

    predictions: list[functional.PredictedToken]


@dataclass(frozen=True, kw_only=True)
class LinearRelationOutput(RelationOutput):
    """Relation output, the input `h`, and the predicted object hidden state `z`."""

    h: torch.Tensor
    z: torch.Tensor

    def as_relation_output(self) -> RelationOutput:
        return RelationOutput(predictions=self.predictions)


@dataclass(frozen=True, kw_only=True)
class RelationOperator:
    """An abstract relation operator, which maps subjects to objects."""

    def __call__(self, subject: str, **kwargs: Any) -> RelationOutput:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    """A linear approximation of a relation inside an LM."""

    mt: models.ModelAndTokenizer
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    h_layer: Layer
    z_layer: Layer
    prompt_template: str
    beta: float | None = None
    metadata: dict = field(default_factory=dict)

    def __call__(
        self,
        subject: str,
        k: int = 5,
        h: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationOutput:
        """Predict the top-k objects for a given subject.

        Args:
            subject: The subject.
            prompt_template: Override for the default prompt template.
            k: Number of objects to return.
            h: Precomputed h, if available.

        Returns:
            Predicted objects and some metadata.

        """
        if kwargs:
            raise ValueError(f"unexpected kwargs: {kwargs}")

        if h is None:
            prompt = functional.make_prompt(
                mt=self.mt, prompt_template=self.prompt_template, subject=subject
            )
            logger.debug(f'computing h from prompt "{prompt}"')

            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt, prompt=prompt, subject=subject
            )

            [[hs], _] = functional.compute_hidden_states(
                mt=self.mt, layers=[self.h_layer], inputs=inputs
            )
            h = hs[:, h_index]
        else:
            logger.debug("using precomputed h")

        z = h
        if self.weight is not None:
            z = z.mm(self.weight.t())
        if self.bias is not None:
            bias = self.bias
            if self.beta is not None:
                z = z * self.beta  # scaling the contribution of Jh with beta
            z = z + bias

        lm_head = self.mt.lm_head if not self.z_layer == "ln_f" else self.mt.lm_head[:1]
        logits = lm_head(z)
        dist = torch.softmax(logits.float(), dim=-1)

        topk = dist.topk(dim=-1, k=k)
        probs = topk.values.view(k).tolist()
        token_ids = topk.indices.view(k).tolist()
        words = [self.mt.tokenizer.decode(token_id) for token_id in token_ids]

        return LinearRelationOutput(
            predictions=[
                functional.PredictedToken(token=w, prob=p) for w, p in zip(words, probs)
            ],
            h=h,
            z=z,
        )


@dataclass(frozen=True, kw_only=True)
class LinearRelationEstimator:
    """Abstract method for estimating a linear relation operator."""

    mt: models.ModelAndTokenizer

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class JacobianEstimator(LinearRelationEstimator):
    """Estimate a linear relation operator as a first-order approximation."""

    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(samples=relation.samples, prompt_templates=relation.prompt_templates)

        subject = relation.samples[0].subject
        prompt_template = relation.prompt_templates[0]
        return self.estimate_for_subject(subject, prompt_template)

    def estimate_for_subject(
        self, subject: str, prompt_template: str
    ) -> LinearRelationOperator:
        prompt = functional.make_prompt(
            mt=self.mt, prompt_template=prompt_template, subject=subject
        )
        logger.debug("estimating J for prompt:\n" + prompt)

        h_index, inputs = functional.find_subject_token_index(
            mt=self.mt, prompt=prompt, subject=subject
        )
        logger.debug(f"note that subject={subject}, h_index={h_index}")

        approx = functional.order_1_approx(
            mt=self.mt,
            prompt=prompt,
            h_layer=self.h_layer,
            h_index=h_index,
            z_layer=self.z_layer,
            z_index=-1,
            inputs=inputs,
        )
        return LinearRelationOperator(
            mt=self.mt,
            weight=approx.weight,
            bias=approx.bias,
            h_layer=approx.h_layer,
            z_layer=approx.z_layer,
            prompt_template=prompt_template,
            beta=self.beta,
            metadata=approx.metadata,
        )


@dataclass(frozen=True)
class JacobianIclEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        train = relation.samples[0]
        examples = relation.samples[1:]
        prompt_template = relation.prompt_templates[0]
        prompt_template_icl = functional.make_prompt(
            mt=self.mt, prompt_template=prompt_template, examples=examples, subject="{}"
        )

        # NB(evan): Composition, not inheritance.
        return JacobianEstimator(
            mt=self.mt,
            h_layer=self.h_layer,
            z_layer=self.z_layer,
            beta=self.beta,
        ).estimate_for_subject(train.subject, prompt_template_icl)


from src import lens


@dataclass(frozen=True)
class JacobianIclMeanEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None
    rank: int | None = None  # If None, don't do low rank approximation.

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)

        samples = relation.samples
        prompt_template = relation.prompt_templates[0]

        approxes = []
        for sample in samples:
            prompt = functional.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject=sample.subject,
                examples=samples,
            )
            logger.debug("estimating J for prompt:\n" + prompt)

            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
            )
            logger.debug(f"note that subject={sample.subject}, h_index={h_index}")

            approx = functional.order_1_approx(
                mt=self.mt,
                prompt=prompt,
                h_layer=self.h_layer,
                h_index=h_index,
                z_layer=self.z_layer,
                z_index=-1,
                inputs=inputs,
            )
            approxes.append(approx)

        weight = torch.stack([approx.weight for approx in approxes]).mean(dim=0)
        bias = torch.stack([approx.bias for approx in approxes]).mean(dim=0)

        # TODO(evan): J was trained on with N - 1 ICL examples. Is it a
        # problem that the final prompt has N? Probably not, but should test.
        prompt_template_icl = functional.make_prompt(
            mt=self.mt,
            prompt_template=prompt_template,
            examples=samples,
            subject="{}",
        )

        if self.rank is not None:
            weight = functional.low_rank_approx(matrix=weight, rank=self.rank)

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=approxes[0].z_layer,
            prompt_template=prompt_template_icl,
            beta=self.beta,
            metadata={
                "Jh": [approx.metadata["Jh"].squeeze() for approx in approxes],
                # "approxes": approxes,
            },
        )

        return operator


@dataclass(frozen=True)
class JacobianIclMeanEstimator_Imaginary(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None
    rank: int | None = None  # If None, don't do low rank approximation.
    interpolate_on: int = 2  # number of examples to average on to get the imaginary h
    n_trials: int = 5  # (maximum) number of trials to average over
    average_on_sphere: bool = True  # will interpolate to make all latent vectors have the same norm (hence contribution?)
    magnitude_h: float | None = None  # ||h_myth||, if average_on_sphere is True. Shouldn't matter much, since `o` should be insensitive to `||h||` anyways
    assert (
        interpolate_on >= 2
    ), """need at least 2 examples to get imaginary latent. 
    Call JacobianIclMeanEstimator to calculate on real h instead"""

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)

        samples = relation.samples
        n_icl = len(samples) - self.interpolate_on
        if n_icl < 3:
            logger.warning(
                f"Number of free examples is {n_icl}. It is recommended to have at least 3."
            )
        prompt_template = relation.prompt_templates[0]

        approxes = []
        candidate_combinations = list(
            itertools.combinations(samples, self.interpolate_on)
        )
        random.shuffle(candidate_combinations)
        for interpolation_candidates in candidate_combinations[
            : min(self.n_trials, len(candidate_combinations))
        ]:
            logger.debug(
                f"interpolation candidates: {', '.join([candidate.__str__() for candidate in interpolation_candidates])}"
            )
            # use all other examples as few-shot
            icl_examples = list(set(samples) - set(interpolation_candidates))

            prompt = functional.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=icl_examples,
            )
            logger.debug("estimating J for prompt:\n" + prompt)

            # use the first subject to get h_index
            s1 = interpolation_candidates[0].subject
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt.format(s1),
                subject=s1,
            )
            logger.info(f"note that subject={s1}, h_index={h_index}")

            candidate_hs = functional.compute_hs_and_zs(
                mt=self.mt,
                prompt_template=prompt_template,
                subjects=[candidate.subject for candidate in interpolation_candidates],
                h_layer=self.h_layer,
                z_layer=self.z_layer,
                examples=icl_examples,
            ).h_by_subj

            if self.average_on_sphere:
                if self.magnitude_h is None:
                    l2_norm = (
                        torch.stack([h for h in candidate_hs.values()])
                        .mean(dim=0)
                        .norm()
                    )
                else:
                    l2_norm = self.magnitude_h
                logger.info(f"{l2_norm=:.3f}")
                for subj in candidate_hs.keys():
                    candidate_hs[subj] = (candidate_hs[subj] * l2_norm) / candidate_hs[
                        subj
                    ].norm()

            for subj, h in candidate_hs.items():
                logger.debug(f"{subj=} | h_norm={h.norm().item()}")

            mythical_h = torch.stack([h for h in candidate_hs.values()]).mean(dim=0)
            logger.debug(f"mythical_h_norm={mythical_h.norm().item()}")

            approx = functional.order_1_approx(
                mt=self.mt,
                prompt=prompt.format(s1),
                h_layer=self.h_layer,
                h_index=h_index,
                z_layer=self.z_layer,
                z_index=-1,
                h=mythical_h,
                inputs=inputs,
            )
            approxes.append(approx)
            logger.debug("----------------------------------")

        weight = torch.stack([approx.weight for approx in approxes]).mean(dim=0)
        bias = torch.stack([approx.bias for approx in approxes]).mean(dim=0)

        prompt_template_icl = functional.make_prompt(
            mt=self.mt,
            prompt_template=prompt_template,
            examples=samples,
            subject="{}",
        )

        if self.rank is not None:
            weight = functional.low_rank_approx(matrix=weight, rank=self.rank)

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=approxes[0].z_layer,
            prompt_template=prompt_template_icl,
            beta=self.beta,
            metadata={
                "Jh": [approx.metadata["Jh"].squeeze() for approx in approxes],
                # "approxes": approxes,
            },
        )

        return operator


@dataclass(frozen=True, kw_only=True)
class CornerGdEstimator(LinearRelationEstimator):
    """Estimate a "corner" of LM's rep space where range is assigned equal prob."""

    h_layer: Layer

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        result = functional.corner_gd(mt=self.mt, words=list(relation.range))
        return LinearRelationOperator(
            mt=self.mt,
            weight=None,
            bias=result.corner,
            h_layer=self.h_layer,
            z_layer=-1,
            prompt_template="{}",
        )


@dataclass(frozen=True)
class LearnedLinearEstimatorBaseline(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    mode: Literal["zs", "icl"] = "zs"
    n_steps: int = 100
    lr: float = 5e-2
    weight_decay: float = 2e-2

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        device = models.determine_device(self.mt)
        dtype = models.determine_dtype(self.mt)
        samples = relation.samples
        prompt_template = (
            self.mt.tokenizer.eos_token + " {}"
            if self.mode == "zs"
            else relation.prompt_templates[0]
        )

        H_stack: list[torch.Tensor] = []
        Z_stack: list[torch.Tensor] = []

        if self.z_layer is None:
            z_layer = models.determine_layers(self.mt)[-1]

        h_layer_name, z_layer_name = models.determine_layer_paths(
            self.mt, [self.h_layer, z_layer]
        )

        for sample in samples:
            if self.mode == "zs":
                prompt = prompt_template.format(sample.subject)
            elif self.mode == "icl":
                prompt = functional.make_prompt(
                    mt=self.mt,
                    prompt_template=prompt_template,
                    subject=sample.subject,
                    examples=samples,
                )
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
            )

            with baukit.TraceDict(
                self.mt.model,
                [h_layer_name, z_layer_name],
            ) as traces:
                output = self.mt.model(**inputs)

            H_stack.append(
                functional.untuple(traces[h_layer_name].output)[0][h_index].detach()
            )
            Z_stack.append(
                functional.untuple(traces[z_layer_name].output)[0][-1].detach()
            )

        H = torch.stack(H_stack, dim=0).to(torch.float32)
        Z = torch.stack(Z_stack, dim=0).to(torch.float32)

        n_embd = models.determine_hidden_size(self.mt)
        weight = torch.empty(n_embd, n_embd, device=device)
        bias = torch.empty(1, n_embd, device=device)
        weight.uniform_(-0.1, 0.1)
        bias.uniform_(-0.1, 0.1)
        weight.requires_grad = True
        bias.requires_grad = True

        optimizer = torch.optim.Adam(
            [weight, bias], lr=self.lr, weight_decay=self.weight_decay
        )

        for _ in range(self.n_steps):
            Z_hat = H.mm(weight.t()) + bias
            loss = (Z - Z_hat).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.mode == "icl":
            prompt_template = functional.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=samples,
            )

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight.detach().to(dtype).to(device),
            bias=bias.detach().to(dtype).to(device),
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )

        return operator


@dataclass(frozen=True)
class OffsetEstimatorBaseline(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    scaling_factor: float | None = None
    mode: Literal["icl", "zs"] = "icl"

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)

        prompt_template = (
            self.mt.tokenizer.eos_token + " {}"
            if self.mode == "zs"
            else relation.prompt_templates[0]
        )

        range_tokenized = models.tokenize_words(
            tokenizer=self.mt.tokenizer, words=list(relation.range)
        )
        range_tokenized = [t[0].item() for t in range_tokenized.input_ids]

        unembedding_rows = self.mt.lm_head[1].weight[range_tokenized]
        unembedding_rows = torch.stack(
            [row / row.norm() for row in unembedding_rows]
        )  # so that all of the embeddings are unit vectors
        offset = unembedding_rows.mean(dim=0)[None]

        if self.scaling_factor is None:
            H = []
            h_layer_name = models.determine_layer_paths(self.mt, [self.h_layer])[0]
            training_samples = (
                relation.samples if len(relation.samples) < 8 else relation.samples[:8]
            )
            for sample_idx in range(len(training_samples)):
                sample = training_samples[sample_idx]
                if self.mode == "zs":
                    prompt = prompt_template.format(sample.subject)
                elif self.mode == "icl":
                    prompt = functional.make_prompt(
                        mt=self.mt,
                        prompt_template=prompt_template,
                        subject=sample.subject,
                        examples=training_samples[0:sample_idx]
                        + training_samples[sample_idx + 1 :],
                    )
                h_index, inputs = functional.find_subject_token_index(
                    mt=self.mt,
                    prompt=prompt,
                    subject=sample.subject,
                )

                with baukit.TraceDict(
                    self.mt.model,
                    [h_layer_name],
                ) as traces:
                    output = self.mt.model(**inputs)

                H.append(
                    functional.untuple(traces[h_layer_name].output)[0][h_index].detach()
                )

            h_mean = torch.stack(H, dim=0).mean(dim=0)
            scaling_factor = h_mean.norm() / offset.norm()
            scaling_factor /= 2
        else:
            scaling_factor = self.scaling_factor

        offset = offset * scaling_factor

        if self.z_layer is None:
            z_layer: Layer = models.determine_layers(self.mt)[-1]
        else:
            z_layer = self.z_layer

        if self.mode == "icl":
            prompt_template = functional.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=training_samples,
            )

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=None,
            bias=offset,
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )

        return operator


def _check_nonempty(**values: list) -> None:
    for key, value in values.items():
        if len(value) == 0:
            raise ValueError(f"expected at least one value for {key}")


def _warn_gt_1(**values: list) -> None:
    for key, value in values.items():
        if len(value) > 1:
            logger.warning(f"relation has > 1 {key}, will use first ({value[0]})")
