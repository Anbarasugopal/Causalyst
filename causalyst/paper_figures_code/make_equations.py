"""
make_equations.py

Renders equation images for the paper's Methods section. Deliberately
LIMITED to components that are actually implemented and run:
    (1) Market graph formalism        -- graph_builder.py, real
    (2) Granger causality F-test      -- causal_discovery.py, real
    (3) PCMCI / MCI test              -- pcmci_discovery.py, real
    (4) Synthesizer confidence rule   -- agents.py PROMPT SPEC (design,
                                          not yet empirically validated --
                                          labeled as such in the paper)

Deliberately NOT rendering: any equation for the temporal GNN predictor,
since that component does not exist in code yet. Adding equations for an
unbuilt component would dress up something fabricated -- exactly what
this project has been avoiding throughout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"


def render(latex, filename, fontsize=15, figsize=(6.5, 0.9)):
    fig = plt.figure(figsize=figsize)
    fig.text(0.02, 0.5, latex, fontsize=fontsize, va="center", ha="left")
    plt.axis("off")
    plt.savefig(f"/home/claude/causalyst/paper/figures/{filename}", dpi=220,
                bbox_inches="tight", pad_inches=0.08, transparent=False)
    plt.close()
    print(f"saved {filename}")


# (1) Market graph formalism
render(
    r"$G = (V, E), \quad V = V_{company} \cup V_{sector} \cup V_{flow} \cup V_{macro}, "
    r"\quad E \subseteq V \times V \times \{causal,\ sector,\ ownership\}$",
    "eq1_graph.png", fontsize=13, figsize=(7.2, 0.8),
)

# (2) Granger causality F-test
render(
    r"$H_0: \ y_t = \sum_{k=1}^{p} \alpha_k y_{t-k} + \varepsilon_t \quad$ vs $\quad "
    r"H_1: \ y_t = \sum_{k=1}^{p} \alpha_k y_{t-k} + \sum_{k=1}^{p} \beta_k x_{t-k} + \varepsilon_t$",
    "eq2_granger.png", fontsize=14, figsize=(7.4, 0.9),
)
render(
    r"$x \rightarrow y\ \ (\text{Granger-causal}) \ \Leftrightarrow\ \text{F-test rejects } H_0"
    r"\text{ at } \alpha_{corr} = \alpha \,/\, [n(n{-}1)]\ \ (\text{Bonferroni}, n \text{ vars})$",
    "eq2b_granger_test.png", fontsize=13, figsize=(7.4, 0.8),
)

# (3) PCMCI / MCI conditional independence
render(
    r"$X^i_{t-\tau} \perp X^j_t \ \ |\ \ \mathcal{P}(X^j_t)\setminus X^i_{t-\tau},\ \ "
    r"\mathcal{P}(X^i_{t-\tau}) \quad \Rightarrow\quad \text{tested via partial correlation}$",
    "eq3_pcmci.png", fontsize=13, figsize=(7.4, 0.8),
)

# (4) Synthesizer confidence rule (design specification, not yet validated empirically)
render(
    r"$Conf_{final} = \left(\frac{1}{4}\sum_{i=1}^{4} Conf_i\right) \times "
    r"\left(1 - \lambda \cdot D(agents)\right), \quad D(agents) = \text{pairwise stance disagreement}$",
    "eq4_synthesis.png", fontsize=13, figsize=(7.4, 0.8),
)

print("done")
