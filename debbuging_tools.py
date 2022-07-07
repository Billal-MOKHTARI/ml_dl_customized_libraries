import matplotlib.pyplot as plt

def learning_curve(history, figsize=(8, 8), debug_type = 'both'):
    hist = history

    if debug_type == 'loss' or debug_type == 'both':
        plt.figure(figsize=figsize)
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Training cost', 'Validation cost'])

    if debug_type == 'accuracy' or debug_type == 'both':
        plt.figure(figsize=figsize)
        plt.plot(hist['accuracy'])
        plt.plot(hist['val_accuracy'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Training accuracy', 'Validation accuracy'])
