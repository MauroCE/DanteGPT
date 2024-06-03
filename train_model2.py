import time
import torch
import pickle
from gpt import Model2, estimate_loss, get_batch
from gpt_configurations import GPTConfig2


if __name__ == "__main__":
    torch.manual_seed(333)  # reproducibility

    # Settings
    config = GPTConfig2()
    config.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    max_iters = 10000
    eval_interval = 500
    learning_rate = 1e-2  # This learning rate seems good
    eval_iters = 200

    # Device (this works for mac silicons, use cuda for nvidia gpus)
    print("DEVICE: ", config.device)

    # Read divina commedia
    with open('data/commedia.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Compute vocabulary size for divina commedia, here we work on a character level
    vocabulary = sorted(list(set(text)))
    vocab_size = len(vocabulary)

    # Mappings from characters to integers and vice versa
    str_to_int = {character: integer for integer, character in enumerate(vocabulary)}
    int_to_str = {integer: character for integer, character in enumerate(vocabulary)}

    # Encoder and Decoder from string to indices and vice versa
    str2int = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
    int2str = lambda int_list: ''.join([int_to_str[integer] for integer in int_list])  # list(int) --> string

    # Encode divina commedia
    data = torch.tensor(str2int(text), dtype=torch.long)

    # (Naive) Train-Test split
    n = int(0.9*len(data))
    train_data = data[:n]  # 90% training
    val_data = data[n:]    # 10% validation

    # Instantiate model and send params to device
    model = Model2(config)
    gpt = model.to(config.device)

    # Adam optimizer, as usual
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)

    # Store losses
    training_losses = []
    validation_losses = []

    # Store initial time
    start_time = time.time()

    # Training loop
    for iteration in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iteration % eval_interval == 0:
            losses = estimate_loss(
                gpt_model=model,
                training_data=train_data,
                dev=config.device,
                validation_data=val_data,
                eval_iters=eval_iters, context_size=config.context_size,
                batch_size=config.batch_size)
            print(f"step {iteration} train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")
            validation_losses.append(losses['val'])
            print("\tTime passed: ", time.time() - start_time)

        # sample a batch of data
        xb, yb = get_batch(split="train",
                           training_data=train_data,
                           validation_data=val_data,
                           dev=config.device,
                           context_size=config.context_size, batch_size=config.batch_size)

        # evaluate the loss
        logits, loss = model(idx=xb, device=config.device, targets=yb)
        training_losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model.state_dict(), "models/model2_smaller2.pth")
    with open("losses/model2_smaller2_training_{}.pkl".format(eval_interval), "wb") as file:
        pickle.dump(training_losses, file)
    with open("losses/model2_smaller2_validation_{}.pkl".format(eval_interval), "wb") as file:
        pickle.dump(validation_losses, file)

    # Save final time
    total_time = time.time() - start_time
    print("Total time: ", total_time)
    with open("timings/model2_smaller2_{}.pkl".format(eval_interval), "wb") as file:
        pickle.dump([total_time], file)
