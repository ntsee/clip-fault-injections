import json
import utils
import matplotlib.pyplot as plt


def create_embedding_title(encoder, embedding, fault_length):
    encoder = encoder.replace("_", " ").replace("model", "encoder").title()
    embedding = embedding.replace("_", " ").title()
    return f'{encoder} {embedding} Activation Faults (fault_length={fault_length * 100}%) on CIFAR10'


def create_encoder_title(encoder, encoder_block_index, fault_length):
    encoder = encoder.replace("_", " ").replace("model", "encoder").title()
    return f'{encoder} MLP Block {encoder_block_index} Activation Faults (fault_length={fault_length * 100}%) on CIFAR10'


def create_graph(title, data):
    plt.figure(figsize=(7.5, 7.5))
    plt.title(title)
    plt.xlabel('Fault Offset %')
    plt.ylabel('Accuracy')
    plt.ylim(0, 0.9)

    cm = plt.get_cmap('tab20b')
    plt.plot(data[0], data[1], marker='o', color=cm(0))

    plt.savefig(f'activation_graphs/{title}.png')
    plt.show()


def main():
    embeddings = {
        'text_model': ['position_embedding', 'token_embedding'],
        'vision_model': ['position_embedding', 'patch_embedding']
    }

    core_layers = ['fc1', 'fc2', 'activation_fn']

    for fault_length in [0.01]:
        '''
        for core_layer in core_layers:
            x = list(range(100))
            y = []
            for offset in x:
                file_name = f'activations/vision_model.encoder.layers.0.mlp.{core_layer}_{offset}.json'
                with utils.safe_open(file_name, 'r') as f:
                    results = json.load(f)
                    accuracy = results['accuracy']
                    y.append(accuracy)

            title = f'Vision Encoder MLP 0 {core_layer.upper()} Activation Faults (fault_length={fault_length * 100}%) on CIFAR10'
            create_graph(title, (x, y))
        '''

        for encoder in ['text_model', 'vision_model']:
            '''
            for embedding in embeddings[encoder]:
                x = list(range(100))
                y = []
                for offset in x:
                    file_name = f'activations/{encoder}_{embedding}_{fault_length}_{offset}.json'
                    with utils.safe_open(file_name, 'r') as f:
                        results = json.load(f)
                        accuracy = results['accuracy']
                        y.append(accuracy)

                title = create_embedding_title(encoder, embedding, fault_length)
                create_graph(title, (x, y))
            '''

            #'''
            for encoder_block_index in [0, 5, 8]:
                x = list(range(100))
                y = []
                for offset in x:
                    file_name = f'activations/{encoder}_{encoder_block_index}_{fault_length}_{offset}.json'
                    with utils.safe_open(file_name, 'r') as f:
                        results = json.load(f)
                        accuracy = results['accuracy']
                        y.append(accuracy)

                title = create_encoder_title(encoder, encoder_block_index, fault_length)
                create_graph(title, (x, y))
            #'''



if __name__ == '__main__':
    main()
