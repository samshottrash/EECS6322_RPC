from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    Pipeline,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
    sweep,
)

'''
Also need to add approprate comments from notebook

Also beginning to think, might just have to implement our own sae????
cause the sae library is for texts, not images. Hmm... not sure tbh
Dont think it would be too hard tho
'''

# from config import clip_model_name

def get_sweep_config(clip_model_name):
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(values=[3e-5, 1.5e-4, 3e-4, 1.5e-3, 3e-3])
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
            ),
            # source_model=SourceModelHyperparameters(
            #    name=Parameter("openai/clip"),
            #   cache_names=Parameter(["vision_model.encoder.layers.11"]),
            #    hook_dimension=Parameter(768 if clip_model_name == "ViT-B/16" else 1024)
            # ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("cc3m_clip_features"),
                context_size=Parameter(256),
                pre_tokenized=Parameter(False),
                pre_download=Parameter(False),
                tokenizer_name=Parameter("openai/clip-vit-base-patch32")
            ),
            autoencoder=AutoencoderHyperparameters(
                expansion_factor=Parameter(values=[2, 4, 8])
            ),
            # num_epochs=Parameter(200),
            #resample_interval=Parameter(10)
        ),
        method=Method.RANDOM
    )
    # sweep(sweep_config=sweep_config)

    return sweep_config

def train_autoencoder(clip_model_name):

    '''
    changed it from pipeline to their example in the colab
    Cause their docs said: Pipeline for training a Sparse Autoencoder on TransformerLens activations
    And I dont think we are using activations from TLens?????
    Ask Samiha why pipeline
    '''

    # pipeline = Pipeline(sweep_config)
    # num_neurons_fired = pipeline.train_autoencoder()

    sweep_config = get_sweep_config(clip_model_name)
    sweep(sweep_config=sweep_config)
