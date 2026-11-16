"""Minimal Streamlit front end. Run: streamlit run app/dashboard.py"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.core import MLP
from src.core.activations import ACTIVATIONS
from src.experiments.xor import make_xor

st.set_page_config(page_title="Neural Network Lab", layout="wide")
st.title("Neural Network Lab")
st.caption("A transparent multilayer perceptron built with NumPy — inspect the mathematics, not just the prediction.")

tab1, tab2, tab3 = st.tabs(["Architecture & forward pass", "Activation functions", "XOR experiment"])

with tab1:
    st.markdown("For each layer: **Z = A @ W + b**, then **A_next = g(Z)**.")
    sizes = st.text_input("Layer sizes (comma-separated)", "2,4,1")
    if st.button("Initialize example network"):
        try:
            layer_sizes = [int(v.strip()) for v in sizes.split(",")]
            names = ["tanh"] * (len(layer_sizes) - 2) + ["sigmoid"]
            net = MLP(layer_sizes, names, "binary_cross_entropy")
            sample = np.zeros((1, layer_sizes[0]))
            prediction, cache = net.forward(sample, return_cache=True)
            st.write(f"Parameter count: {net.parameter_count()}")
            for i, (z, a) in enumerate(zip(cache["Z"], cache["A"][1:]), start=1):
                st.write(f"Layer {i}: Z shape {z.shape}, A shape {a.shape}")
                st.code(f"Z = {np.array2string(z, precision=4)}\\nA = {np.array2string(a, precision=4)}")
            st.success(f"Output: {prediction}")
        except Exception as exc:
            st.error(str(exc))

with tab2:
    z = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots()
    for name in ["relu", "sigmoid", "tanh"]:
        ax.plot(z, ACTIVATIONS[name][0](z), label=name)
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel("g(z)")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with tab3:
    st.write("XOR is deliberately small: it demonstrates why a nonlinear hidden layer matters.")
    X, y = make_xor()
    epochs = st.slider("Training epochs", 100, 8000, 3000, 100)
    rate = st.slider("Learning rate", 0.05, 1.0, 0.5, 0.05)
    if st.button("Train XOR network"):
        net = MLP([2, 4, 1], ["tanh", "sigmoid"], "binary_cross_entropy", seed=7)
        history = net.fit(X, y, epochs=epochs, learning_rate=rate,
                          batch_size=4, shuffle=False)
        st.line_chart({"loss": history["loss"]})
        st.write("Predictions (in input-row order):")
        st.dataframe({"x1": X[:, 0], "x2": X[:, 1],
                      "target": y[:, 0], "probability": net.predict(X)[:, 0]})
