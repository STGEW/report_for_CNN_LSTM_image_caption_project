import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        # embedding layer, convert our vocab to the vector of features
        self.embed = nn.Embedding(vocab_size, embed_size)

        # lstm, expects input - a feature vector, output - hidden size
        # batch_first=True <- to expect first dimension in tensor to be a batch number
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        # A linear layer to convert lstm's output to our vocab
        self.lstm_out_to_vocab = nn.Linear(hidden_size, vocab_size)


    def forward(self, features, captions):

        # Convert our captions from text to vector of features
        embeds = self.embed(captions[:, :-1])

        # unsqueeze the CNN output
        features = features.unsqueeze(dim=1)

        # stack together CNN output + captions
        inp = torch.cat((features, embeds), dim=1)

        # feed lstm with a full input
        out, hid = self.lstm(inp)

        # convert lstm's output to the vocab output
        out = self.lstm_out_to_vocab(out)

        return out


    def sample(self, inputs, states=None, max_len=20):
        # initialize the state of the lstm with a random values
        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
                   torch.randn(1, 1, self.hidden_size).to(inputs.device))

        # our output caption
        captions = []

        # We get only max_len number of words
        for i in range(max_len):

            # feed lstm with iputs and states.
            # Initially states are random values
            lstm_out, states = self.lstm(inputs, states)

            # get scores for next word
            scores = self.lstm_out_to_vocab(lstm_out)

            # squeeze dimension
            scores = scores.squeeze(1)

            # get the most probable word
            word = scores.argmax(dim=1)
            captions.append(word.item())

            # update the inputs
            inputs = self.embed(word.unsqueeze(0))

        return captions
