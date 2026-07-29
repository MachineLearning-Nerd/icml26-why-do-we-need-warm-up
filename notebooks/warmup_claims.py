import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Why Do We Need Warm-up? An exact claim audit

    ![Campaign verdicts](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/main/reports/warmup_claims/images/headline_verdicts.svg)

    The paper models curvature as

    \[
    \|\nabla^2 f(w)\| \le H_0 + H_1(f(w)-f^\star).
    \]

    This notebook opens with the already-computed evidence. It does not
    rerun the expensive FineWeb or ImageNet32 routes.
    """)
    return


@app.cell
def _():
    claims = {
        "Claim 1": {
            "status": "VERIFIED",
            "paper": "Definition 3.1 and smooth-class inclusion",
            "result": "Exact algebra; HVP/full-Hessian relative error 8.95e-11.",
            "scope": "Proposition B.2 needs corrected sum constants.",
        },
        "Claim 2": {
            "status": "FALSIFIED",
            "paper": "Proposition 3.2(ii), strong balancedness",
            "result": "At zero loss gap, Hessian norm is 2t² and is unbounded.",
            "scope": "A missing full-row-rank assumption enables the counterexample.",
        },
        "Claim 3": {
            "status": "FALSIFIED",
            "paper": "Proposition 3.3(ii), Equation (31)",
            "result": "Complete Hessian 63.1359 > maximum printed RHS 4.0883.",
            "scope": "A corrected H0 with the omitted Clinear·f* term is not contradicted.",
        },
        "Claim 4": {
            "status": "FALSIFIED",
            "paper": "Theorem 4.1(3), Equation (63) cap",
            "result": "Observed first hit 1 < claimed lower bound 1.7463.",
            "scope": "Theorem 4.2's adaptive upper bound remains intact.",
        },
        "Claim 5": {
            "status": "FALSIFIED",
            "paper": "Theorem 4.3 displayed formula",
            "result": "Valid PL instance yields a −13.8629 iteration upper bound.",
            "scope": "A corrected small-epsilon recurrence is not contradicted.",
        },
        "Claim 6": {
            "status": "BLOCKED",
            "paper": "Section 3.2 named models and paper-scale early training",
            "result": "Exact parameter counts run, but no short FineWeb slope passed.",
            "scope": "Batch, sequence, tokenizer, precision, and horizon substitutions are material.",
        },
    }
    return (claims,)


@app.cell
def _(claims, mo):
    claim_picker = mo.ui.dropdown(
        options=list(claims),
        value="Claim 2",
        label="Inspect a claim",
    )
    claim_picker
    return (claim_picker,)


@app.cell
def _(claim_picker, claims, mo):
    selected = claims[claim_picker.value]
    mo.md(
        f"""
        ## {claim_picker.value}: {selected["status"]}

        **Paper statement.** {selected["paper"]}

        **Observed evidence.** {selected["result"]}

        **Scope and limitation.** {selected["scope"]}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Two exact counterexamples

    ### Balanced deep linear network

    With \(X=\operatorname{diag}(1,0)\), \(Y=0\), and
    \(W_1=W_2=\operatorname{diag}(0,t)\), balancedness is exact and the
    loss gap is zero. A complete Hessian has norm \(2t^2\), so no finite
    \(H_0\) can satisfy the proposed bound for every \(t\).

    ### Class-stable fixed-step lower bound

    Use \(f(w)=w^2\) on \(|w|\le1\) and
    \(f(w)=2e^{|w|-1}-1\) outside. This is \(C^2\), strongly convex, PL,
    and \((H_0,H_1)\)-smooth with \(H_0=2,H_1=1\). An admissible fixed
    step reaches the minimizer in one iteration where Theorem 4.1(3)
    claims at least 1.7463.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## What remains

    Claim 6 is not resolved by exact parameter counts alone. A faithful
    test still needs the paper's tokenizer and estimator implementation,
    batch 256, sequence lengths 1024/2048, billion-token LM horizons,
    full ImageNet32 schedules, and the reported FP16 setup.

    Run locally:

    ```text
    uv run marimo edit notebooks/warmup_claims.py
    uv run marimo run notebooks/warmup_claims.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
