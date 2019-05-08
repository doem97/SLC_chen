# SLC_chen Introduction

SLC is an Image Classification framework for large-scale DL experiments. It feautres in simple usage and convinient log tracking.

# Brief Summary

The main idea of the project is that the deep-learning experiment include two modules:  _process control_ and _model control_. The two modules are separate and composable. You can do a train as simple as:

```python
pc = ProcessControl('config_file')
mc = ModelControl('config_file')
pc.train(mc)
```

**All-in-one config file.** In above pseudo-code, all parameters are loaded via `config_file`, that means just track the `config_file`, and it will be easy to reproduce the experiment and save huge-scales-parameters.

# Usage

## Pre-Settings

1. Run ./environment_set.sh in the path where you want the SLC_chen project be placed.

    E.g. in path `home/user/`, the script will:
    
    1. create file structure into `home/user/SLC_chen/`
    2. download and unzip the dataset to `home/user/SLC_chen/dataset/128_128`

       **notice: 
        - You can choose what dataset size to use in script
        - Original images didn't provided with download link cause its too huge

2. Move into dir `SLC_chen/experience_space/`, and do experiences.

## SkinLesionClassify()

Class `SkinLesionClassify()` is for _process control_. The class takes control of high level oprations in deep learning experiments.

Class `SkinLesionClassify` control the below process:

- Prepare data
- Set workpaths
- Data generate
- Feed into training or validation
- Get scores
- Save necessary intermediate data
- Save checkpoint
- Create log

The whole training process is as simple as

```python
import SkinLesionClassify
slc = SkinLesionClassify.SkinLesionClassify()
slc.train()
```

But in the process, there are so many parameters that too trivial to log. Thus for control and reproduce, all configurable parameters are loaded from `.conf` file when init the `SkinLesionClassify` class like this:

**notice: `.conf` file must be provided.

```python
config_file = "./SLC.conf"
slc = SkinLesionClassify.SkinLesionClassify(config_file)
```

You can take a look of `.conf` file in `code/SLC_template.conf`. It listed all parameters can be changed.

## SLCModel

Class `SLCModel()` is for _model control_. The class has a Keras.model() object as main part.

`SLCModel()` can do:
- Load metrics and loss
- Compile Keras model object
- Change model

The model control is as:

```python
import SLC_models
slc_model = SLC_models.SLCModel('config_file')
slc_model.loadModel('model_name', **kwargs)
```

For elastic change of model object, here is the only `**kwargs` didn't zipped in config file. In one set of similar experiments, the **kwargs may be the most frequently changed parameter, but don't worry, it will be saved into a log file part from `config_file`.

## Example

```python
import SkinLesionClassify
import SLC_models

config_file = "./SLC.conf"
slc = SkinLesionClassify.SkinLesionClassify(config_file)
slc.resizeImages()
slc_model = SLC_models.SLCModel(config_file)
slc_model.loadModel('resnet_v1', depth = 20)
slc.train(slc_model)
slc.evaluation(slc_model)
```

# Notice

- /code folder should be more stastic and .gitignore file will ignore tracking operations in /experience_space
- data files can be orgnized like:
    - /dataset/HAM10000_metadata.csv,/origin,/resized_{}_{}
