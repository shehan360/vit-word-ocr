import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math
import unidecode
from tqdm import tqdm

all_characters = [" "]
all_characters.extend(string.printable[:62])
n_characters = len(all_characters)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    str = ''
    for char in file[start_index:end_index]:
        if char not in all_characters and char != " ":
            continue
        else:
            str += char
    return str

class ShakespereLM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(ShakespereLM, self).__init__()
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

def train_step(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(len(inp) - 1):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, torch.unsqueeze(target[c], dim=0))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len


if __name__ == '__main__':

    chunk_len = 200

    file = unidecode.unidecode(open('/Users/shehan360/PycharmProjects/vit-ocr/shakespeare_ocr/_corpora/completeworks.txt').read())
    print("File length before processing:", len(file))

    str = ''
    # pre process data
    for char in " ".join(file.split()):
        if char not in all_characters and char != " ":
            continue
        else:
            str += char
    file = str
    file_len = len(file)

    print("File length after processing:", file_len)

    print(random_chunk())

    n_iterations = 50000
    print_every = 1000
    plot_every = 1000
    hidden_size = 100
    n_layers = 5
    lr = 0.005

    decoder = ShakespereLM(n_characters, hidden_size, n_characters, n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    all_losses = []
    loss_avg = 0
    best_loss = float('inf')
    for iteration in tqdm(range(1, n_iterations)):
        loss = train_step(*random_training_set())
        loss_avg += loss

        if iteration % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), iteration, iteration / n_iterations * 100, loss))
            print(evaluate('T', 100), '\n')
            if loss_avg / plot_every < best_loss:
                best_loss = loss_avg / plot_every
                torch.save(decoder.state_dict(), 'lm_shakespere_best.pt')

        if iteration % plot_every == 0:
            print("Loss:", loss_avg / plot_every)
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    evaluate('Th', 100)
    torch.save(decoder.state_dict(), 'lm_shakespere.pt')
