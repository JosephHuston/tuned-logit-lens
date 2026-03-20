# Tuned-Lens-and-Logit-Lens

Implementation of tuned and logit lens

### Authors:
- **Joe Huston**


## How to run the code:
1. Clone the repo.
3. Run `uv sync` in the top level repo directory to create a virtual environment with all the correct dependencies installed, then activate the virtual environment. The project expects Python 3.11 or 3.12.
4. Running `logit_lens.py` with `python logit_lens.py` (or similar, depending on OS/environment) and the same for `tuned_lens.py` will run each code with an input string that is hardcoded in the code saved as `prompt` in the main at the bottom. This will make a plot of the preditions at eavh layer using the lenses, stored in `./plots`

## Discussion on tuned/logic lens as tools
While logit lens and tuned lens can provide some insight into what a model is "thinking", they both have their flaws.
### Logic Lens
The main drawback of logit lens is that earlier layers of the model may be doing something in an entirely different format than the last layer, so directly taking the logit and sending it through the last layer may result in nonsense. Additionally, like mentioned in class, the model stores information in superposition, so collapsing the prediction into 1 token will remove all of the nuance of how the model is thinking.
### Tuned Lens
Tuned lens has the same problem as logic lens where it also removes the nuance of superposition of the model at each layer. Since this is an additional layer that is trained on the data at each layer, it could be making some predictions that perhaps the actual model we are looking at wasn't making, as it is trained to minimize loss. On this same note the model is trained on the final output so even if the layer it is currently looking at was "thinking" of something else, the answer wil ltrend towards the final layer prediction.






