import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math

all_characters = string.printable[:62]
n_characters = len(all_characters)
chunk_len = 6

data = string.printable[:62]
print(data)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def random_chunk():
    start_index = random.randint(0, len(data) - 1)
    if start_index > len(data) - chunk_len:
        rem = start_index + chunk_len - len(data)
        return data[start_index:len(data)] + data[:rem]
    end_index = start_index + chunk_len
    return data[start_index:end_index]

print(random_chunk())

class AbcdLM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(AbcdLM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# Turn string into list of longs
def char_to_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

print(char_to_tensor('abc'))

def random_training_set():
    chunk = random_chunk()
    inp = char_to_tensor(chunk[:-1])
    target = char_to_tensor(chunk[1:])
    return inp, target

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_to_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        softmax = torch.nn.functional.softmax(output)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char)

    return predicted

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len - 1):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, torch.unsqueeze(target[c], dim=0))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len

print(random_training_set())

n_epochs = 500
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

decoder = AbcdLM(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('g', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
evaluate('g', 100)
torch.save(decoder.state_dict(), 'lm_abcd.pt')

