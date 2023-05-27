import logging
from dataclasses import dataclass, field, replace
from typing import Any

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
                bias = self.beta * bias
            z = z + bias

        logits = self.mt.lm_head(z)
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
            metadata={"Jh": [approx.metadata["Jh"].squeeze() for approx in approxes]},
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
class CornerMeanEmbeddingEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    scaling_factor: float | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        samples = relation.samples
        prompt_template = relation.prompt_templates[0]
        range_tokenized = models.tokenize_words(
            tokenizer=self.mt.tokenizer, words=list(relation.range)
        )
        range_tokenized = [t[0].item() for t in range_tokenized.input_ids]

        unembedding_rows = self.mt.lm_head[1].weight[range_tokenized]
        unembedding_rows = torch.stack([row / row.norm() for row in unembedding_rows])
        offset = unembedding_rows.mean(dim=0)[None]

        if self.scaling_factor is None:
            H = []
            h_layer_name = models.determine_layer_paths(self.mt, [self.h_layer])[0]
            for sample in samples:
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
            z_layer = models.determine_layers(self.mt)[-1]

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=None,
            bias=offset,
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )

        return operator


@dataclass(frozen=True)
class Word2VecIclEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)

        samples = relation.samples
        prompt_template = relation.prompt_templates[0]

        z_layer = self.z_layer
        if z_layer is None:
            z_layer = models.determine_layers(self.mt)[-1]

        deltas = []
        for sample in samples:
            examples = [s for s in samples if s != sample]
            hs_by_subj, zs_by_subj = functional.compute_hs_and_zs(
                mt=self.mt,
                prompt_template=prompt_template,
                h_layer=self.h_layer,
                z_layer=self.z_layer,
                examples=examples,
                subjects=[sample.subject],
            )
            delta = (
                zs_by_subj[sample.subject].squeeze()
                - hs_by_subj[sample.subject].squeeze()
            )
            deltas.append(delta)

        bias = torch.stack(deltas, dim=0).mean(dim=0)
        return LinearRelationOperator(
            mt=self.mt,
            weight=None,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )


@dataclass(frozen=True)
class LearnedEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
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
        prompt_template = relation.prompt_templates[0]

        H_stack: list[torch.Tensor] = []
        Z_stack: list[torch.Tensor] = []

        if self.z_layer is None:
            z_layer = models.determine_layers(self.mt)[-1]

        h_layer_name, z_layer_name = models.determine_layer_paths(
            self.mt, [self.h_layer, z_layer]
        )

        for sample in samples:
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

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight.detach().to(dtype).to(device),
            bias=bias.detach().to(dtype).to(device),
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
