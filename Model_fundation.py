import torch
import torch.nn as nn
import torchaudio


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(Wav2Vec2Classifier, self).__init__()

        # Load the pre-trained Wav2Vec2 model from torchaudio
        
        # bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        # bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
        # bundle = torchaudio.pipelines.HUBERT_BASE
        # bundle = torchaudio.pipelines.HUBERT_LARGE
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model()

        # Freeze wav2vec2 layers if you want to only fine-tune the MLP later
        # print("wav2vec2 freezed")
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False
    
        # # Unfreeze the last two layers
        # # Assuming the last two layers are part of the `model.encoder.layers`
        # # Adjust based on the structure of your model
        # for param in self.wav2vec2.encoder.transformer.layers.parameters():
        #     param.requires_grad = True
        
        # Set Wav2Vec2 model to trainable mode
        self.wav2vec2.train()

        # Define an MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(bundle._params['encoder_embed_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        # Extract features using Wav2Vec2
        features, _ = self.wav2vec2(input_values)

        # Use mean pooling over time steps (dim 1)
        pooled_features = features.mean(dim=1)

        # Classify using MLP
        logits = self.classifier(pooled_features)
        return logits, pooled_features
