import argparse

from src import estimate

import torch
import transformers


def run_single_examples(
    *,
    model: estimate.Model,
    tokenizer: estimate.Tokenizer,
    h_layer: int,
    device: estimate.Device,
    k: int,
) -> None:
    # Example 1: "is located in"
    print("--- is located in ---")
    is_located_in, _ = estimate.relation_operator_from_sample(
        model,
        tokenizer,
        "The Space Needle",
        "{} is located in the country of",
        h_layer=h_layer,
        device=device,
    )
    for subject, subject_token_index in (
        ("The Space Needle", -1),
        ("The Eiffel Tower", -2),
        ("The Great Wall", -1),
        ("Niagara Falls", -2),
    ):
        objects = is_located_in(
            subject,
            subject_token_index=subject_token_index,
            device=device,
            return_top_k=k,
        )
        print(f"{subject}: {objects}")

    # Example 2: "is CEO of"
    # This one is less sensitive to which h you choose; can usually just do last.
    print("--- is CEO of ---")
    is_ceo_of, _ = estimate.relation_operator_from_sample(
        model, tokenizer, "Indra Nooyi", "{} is CEO of", h_layer=h_layer, device=device
    )
    for subject in (
        "Indra Nooyi",
        "Sundar Pichai",
        "Elon Musk",
        "Mark Zuckerberg",
        "Satya Nadella",
        "Jeff Bezos",
        "Tim Cook",
    ):
        objects = is_ceo_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")

    # Example 3: "is lead singer of"
    # Seems to *actually* find the "is lead singer of grunge rock group" relation.
    print("--- is lead singer of ---")
    is_lead_singer_of, _ = estimate.relation_operator_from_sample(
        model,
        tokenizer,
        "Chris Cornell",
        "{} is the lead singer of the band",
        h_layer=h_layer,
        device=device,
    )
    for subject in (
        "Chris Cornell",
        "Kurt Cobain",
        "Eddie Vedder",
        "Stevie Nicks",
        "Freddie Mercury",
    ):
        objects = is_lead_singer_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")

    # Example 4: "plays the sport of"
    # Does not work at all. Not sure why.
    print("--- plays the sport of ---")
    plays_sport_of, _ = estimate.relation_operator_from_sample(
        model,
        tokenizer,
        "Megan Rapinoe",
        "{} plays the sport of",
        h_layer=h_layer,
        device=device,
    )
    for subject in (
        "Megan Rapinoe",
        "Larry Bird",
        "John McEnroe",
        "Oksana Baiul",
        "Tom Brady",
        "Babe Ruth",
    ):
        objects = plays_sport_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")


def run_batch_examples(
    *,
    model: estimate.Model,
    tokenizer: estimate.Tokenizer,
    h_layer: int,
    device: estimate.Device,
    k: int,
) -> None:
    for relation, samples, tests in (
        (
            "{} plays the sport of",
            (
                ("Megan Rapinoe", "soccer"),
                ("Larry Bird", "basketball"),
                ("John McEnroe", "tennis"),
            ),
            (
                "Shaquille O'Neal",
                "Babe Ruth",
                "Tom Brady",
                "Tiger Woods",
                "Lionel Messi",
                "Michael Phelps",
                "Serena Williams",
            ),
        ),
        (
            "{} are typically associated with the color",
            (
                ("Bananas", "yellow"),
                ("Blueberries", "blue"),
                ("Strawberries", "red"),
                ("Kiwis", "green"),
            ),
            (
                "Oranges",
                "Carrots",
                "Broccoli",
                "Cotton candy",
                "Apples",
                "Sweet potatoes",
            ),
        ),
        (
            "{} typically work inside of a",
            (
                ("Nurses", "hospital"),
                ("Judges", "courtroom"),
                ("Car mechanics", "garage"),
                ("Farmers", "field"),
                ("Programmers", "office"),
                ("Surgeons", "hospital"),
            ),
            (
                "Chefs",
                "Teachers",
                "Biologists",
                "Bus drivers",
                "Investigators",
                "Policemen",
                "Firefighters",
            ),
        ),
    ):
        operator, metadata = estimate.relation_operator_from_batch(
            model,
            tokenizer,
            samples,
            relation,
            h_layer=h_layer,
            device=device,
        )
        print(f"--- {relation} (J trained on {metadata.subject_for_weight}) ---")
        for subject in tests:
            objects = operator(subject, return_top_k=k, device=device)
            print(f"{subject}: {objects}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="EleutherAI/gpt-j-6B",
        help="language model to use",
    )
    parser.add_argument(
        "--method",
        choices=["single", "batch"],
        default="single",
        help="method for estimating relation operator",
    )
    parser.add_argument("--k", type=int, default=5, help="number of top O's to show")
    parser.add_argument("--layer", type=int, default=15, help="layer to get h from")
    parser.add_argument("--device", help="device to run on")
    args = parser.parse_args()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    h_layer = args.layer
    k = args.k

    print(f"loading {args.model}")
    model_kwargs: dict = {}
    if "gpt-j" in args.model:
        model_kwargs["revision"] = "float16"
        model_kwargs["low_cpu_mem_usage"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, **model_kwargs
    )
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    if args.method == "single":
        run_single_examples(
            model=model, tokenizer=tokenizer, h_layer=h_layer, device=device, k=k
        )
    else:
        assert args.method == "batch"
        run_batch_examples(
            model=model, tokenizer=tokenizer, h_layer=h_layer, device=device, k=k
        )
