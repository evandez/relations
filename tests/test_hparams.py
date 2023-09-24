from pathlib import Path

from src.hparams import RelationHParams


def test_RelationHParams_from_relation_with_non_int_edit_layer():
    hparams = RelationHParams.from_relation("gptj", "name gender")
    assert hparams is not None
    assert hparams.h_layer_edit == "emb"


def test_RelationHParams_can_load_all_relation_files():
    # root dir + /hparams
    hparams_dir = Path(__file__).parent.parent / "hparams"
    models = ["gptj", "llama", "gpt2-xl"]
    for model in models:
        hparams_files = hparams_dir.glob(f"{model}/*.json")
        # just to make sure we're finding real files
        assert len(list(hparams_files)) > 10
        for hparams_file in hparams_files:
            print(hparams_file)
            hparams = RelationHParams.from_json_file(hparams_file)
            assert hparams is not None
            assert hparams.model_name == model
            assert hparams.relation_name.replace(" ", "_") in hparams_file.name
