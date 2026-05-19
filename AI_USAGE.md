# AI Usage Declaration

## Tools Used
- Claude — used for scaffolding, code review, and docstring writing
- GitHub Copilot — used for in-editor autocomplete while writing functions
- ChatGPT — used to expand on financial concepts and cross-check understanding

## How It Was Used
- All AI-generated code was critically evaluated and edited by group members before being incorporated into the final submission. Additionally, AI-generated code was checked against the SciPy reference documentation to ensure accuracy within the code.
- In `src/derivatives/`: Greeks were verified against put-call parity by the team to confirm mathematical correctness, since a function can run without errors while still converging to the wrong value.
- Monte Carlo prices for European options were compared against Black-Scholes analytical prices to confirm the simulation was converging correctly within the expected confidence interval.

## Written Ourselves
- Portfolio composition and option strategy selection: all positions were chosen and justified by the team based on financial analysis.
- All written analysis, discussion, and interpretation throughout the notebooks was written by the group independently. 
- Choice of RBA F2 data as the yield curve source: a team decision based on using real Australian government bond data rather than a generic flat rate.
- The limitations and next steps section of the notebook: entirely written by the group to express the constraints of the prototype.