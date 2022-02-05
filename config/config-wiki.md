# Config files wiki
The configuration files goe

## ModelArgs concept
ModelArgs is a python class in `src/model_args.py`, which holds all the needed python objects (metrics functions, data processor class etc) for the analysis.
The JSON configuration file chooses an instance of this class with the property `model_args`, 
and can also override some of the instance's default properties (by using `model_args_override`).

This make the json configuration a self-contained configuration file, along with the default values placed in `model_args.py`.

## JSON keys documentation
- `model_args`: the name of ModelArgs instance (from `model_args.py`) to define the model's arguments (as part of this, these arguments are served to the model in its `__init__`).
- `model_args_override`: optional overriding of the values of the existing fields in our ModelArgs instance.
- `target_dim`: number of dimensions in the output of the model, e.g.: in VAD regression, target_dim = 3


