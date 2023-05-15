def plot_layer_wise_causal_tracing(causal_tracing_results, title):
    for layer in causal_tracing_results.keys():
        causal_tracing_results[layer] = np.array(causal_tracing_results[layer])
    mean = [causal_tracing_results[layer].mean() for layer in causal_tracing_results.keys()]
    # low = [causal_tracing_results[layer].min() for layer in mt.layer_names]
    # high = [causal_tracing_results[layer].max() for layer in mt.layer_names]

    plt.plot(mean, color="blue", linewidth=3)
    # plt.fill_between(range(len(mean)), low, high, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Layer")
    plt.ylabel("causal_score")
    plt.xticks(range(len(causal_tracing_results.keys()))[::2])
    plt.title(title)

    nrun = causal_tracing_results[list(causal_tracing_results.keys())[0]].shape[0]
    for run in range(nrun):
        arr = []
        for layer in causal_tracing_results.keys():
            arr.append(causal_tracing_results[layer][run])
        plt.plot(arr, alpha=0.2)
    return plt

plot_layer_wise_causal_tracing(cur_relation['causal_tracing'], title = relation_name).show()