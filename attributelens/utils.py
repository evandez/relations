from typing import Any, Sequence, TypeAlias

from src.models import ModelAndTokenizer

import numpy as np
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch


def interpret_logits(
    mt: ModelAndTokenizer,
    logits: torch.Tensor,
    top_k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=top_k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=top_k).values.squeeze().tolist()
    return [
        (mt.tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)
    ]


def logit_lens(
    mt: ModelAndTokenizer,
    h: torch.Tensor,
    interested_tokens: list = [],
    get_proba: bool = False,
) -> tuple[list[tuple[str, float]], dict[int, tuple[float, str]]]:
    logits = mt.lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    candidates = interpret_logits(mt, logits)
    interested_logits = {
        t.item(): (logits[t].item(), mt.tokenizer.decode(t)) for t in interested_tokens
    }
    return candidates, interested_logits


def get_info_for_plotting(
    att_info: dict[str, Any],  # !TODO: attributelens output should have its own type
    layer_skip: int = 0,
    must_have_layers: list[
        int
    ] = [],  # initial layer (0) and final layer (47) will be by default present in must_have_layers
    expected_answers: list[str] = [],
) -> dict[str, Any]:
    v_space_reprs = att_info["v_space_reprs"]
    all_layer_names = list(v_space_reprs[0].keys())
    # y_layer_names = all_layer_names[::layer_skip]
    y_layer_indices = list(range(0, len(all_layer_names), layer_skip))
    final_layer_idx = len(all_layer_names) - 1
    must_have_layers.append(final_layer_idx)

    for layer_idx in must_have_layers:
        if layer_idx not in y_layer_indices:
            y_layer_indices.append(layer_idx)
    y_layer_indices = sorted(y_layer_indices)
    y_layer_names = [all_layer_names[idx] for idx in y_layer_indices]

    start_idx, end_idx = att_info["subject_range"]
    x_tokens = att_info["prompt_tokenized"][start_idx:end_idx]

    confidence_matrix = []
    token_matrix = []
    distribution_matrix_top_k = []

    for layer in y_layer_names:
        confidence_arr = []
        token_arr = []
        distribution_arr_top_k = []

        for token_order in range(len(v_space_reprs)):
            cur_tok = v_space_reprs[token_order][layer][0]
            confidence_arr.append(round(cur_tok[1], 4))

            if cur_tok[0] in expected_answers:
                token_arr.append("<b><i>" + cur_tok[0] + "</i></b>")
            else:
                token_arr.append(cur_tok[0])

            top_k = v_space_reprs[token_order][layer]
            distribution_arr_top_k.append(
                "<br>   ".join([f"'{tup[0]}': {round(tup[1], 6)} " for tup in top_k])
            )

        confidence_matrix.append(confidence_arr)
        token_matrix.append(token_arr)
        distribution_matrix_top_k.append(distribution_arr_top_k)

    return {
        "title": att_info["prompt_template"],
        "y_layer_names": y_layer_names,
        "x_tokens": x_tokens,
        "confidence_matrix": confidence_matrix,
        "token_matrix": token_matrix,
        "top_k": distribution_matrix_top_k,
    }


import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go


def add_rectangle_patches(
    fig: plotly.graph_objects.Figure,
    x: int,
    y: int,
    marker_color: str = "black",
    marker_line_width: int = 2,
) -> None:
    dy = [0, 0.5, 0, -0.5]
    dx = [-0.5, 0, 0.5, 0]

    symbol = [142, 141, 142, 141]
    marker_size = [25, 60] * 2

    for i in range(4):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=[x + dx[i]],
                y=[y + dy[i]],
                marker_symbol=symbol[i],
                marker_color=marker_color,
                marker_line_width=marker_line_width,
                marker_size=marker_size[i],
                hoverinfo="skip",
            )
        )
        fig["data"][-1]["showlegend"] = False


def plot_attribute_lens(
    plotting_info: dict[str, Any],  # output of get_info_for_plotting
    colorscale: str = "reds",
    patch_color: str = "black",
) -> plotly.graph_objects.Figure:
    x_tokens = plotting_info["x_tokens"]
    y_layer_names = plotting_info["y_layer_names"]
    confidence_matrix = plotting_info["confidence_matrix"]
    distribution_matrix_top_k = plotting_info["top_k"]
    token_matrix = plotting_info["token_matrix"]

    # z = plotting_info['confidence_matrix']
    x = list(range(len(x_tokens)))
    y = list(range(len(y_layer_names)))

    z_text = plotting_info["token_matrix"]

    fig = ff.create_annotated_heatmap(
        plotting_info["confidence_matrix"],
        x=x,
        y=y,
        annotation_text=z_text,
        customdata=np.dstack(
            (token_matrix, confidence_matrix, distribution_matrix_top_k)
        ),
        colorscale=colorscale,
    )

    fig.update_traces(
        hovertemplate="<br>".join(
            [
                #   "Token: <b>%{x}</b>",
                #   "Layer: %{y}"
                "<b>'%{customdata[0]}': %{customdata[1]}</b><br>",
                #   "Confidence: %{customdata[1]}%",
                "Top_k:<br>   %{customdata[2]}",
            ]
        )
        + "<extra></extra>"
    )

    # add_rectangle_patches(fig, x = 3, y = 13)
    for token_order in range(len(x_tokens)):
        gen_tok = x_tokens[token_order]
        # print(token_order, " --> ", gen_tok)
        for layer_no in range(len(y_layer_names)):
            cur_tok = token_matrix[layer_no][token_order]
            # print(cur_tok, end = " ")
            # if(cur_tok[0] == gen_tok):
            if "<b><i>" in cur_tok:
                # print("(OK)", end=" ")
                add_rectangle_patches(
                    fig, x=token_order, y=layer_no, marker_color=patch_color
                )
        # print()

    fig.update_layout(yaxis_range=[-0.5, len(y)])
    fig.update_layout(xaxis_range=[-0.5, len(x)])

    fig.update_layout(autosize=False, width=90 * len(x) + 200, height=35 * len(y) + 200)

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=x,
            ticktext=x_tokens,
            tickfont=dict(family="Courier New, Monospace", color="darkblue", size=17),
        )
    )

    fig.update_layout(
        xaxis={"side": "bottom"},
        # yaxis={"side": "right"}
    )

    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=y,
            ticktext=[ny + " " for ny in y_layer_names],
            tickfont=dict(family="verdana", color="firebrick", size=14),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b><i>{plotting_info['title']}</i></b>",
            font=dict(family="Courier New, Monospace", size=20, color="rgb(0,0,0)"),
            y=0.95,
        )
    )

    fig.update_layout(plot_bgcolor="white")

    return fig


def visualize_attribute_lens(
    att_info: dict[str, Any],  # !TODO: attributelens output should have its own type,
    layer_skip: int = 0,  # will skip `layer_skip` number of intermediate layers
    must_have_layers: list[
        int
    ] = [],  # initial layer (0) and final layer (47) will be by default present in must_have_layers
    expected_answers: list[str] = [],
    colorscale: str = "blues",
    patch_color: str = "black",
) -> plotly.graph_objects.Figure:
    print("must_have_layers: ", must_have_layers)
    print("expected_answers: ", expected_answers)
    plotting_info = get_info_for_plotting(
        att_info, layer_skip, must_have_layers, expected_answers
    )
    return plot_attribute_lens(plotting_info, colorscale, patch_color)
