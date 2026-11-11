import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

# Wybór urządzenia
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

# Trening modelu
def train_model(model,train_loader,val_loader,criterion,optimizer,device,epochs=100,patience=10,model_name="model_weights.pth"):
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Pętla treningowa
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # Walidacja modelu
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoka {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        # Sprawdzenie poprawy wyniku walidacji
        if val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Brak poprawy przez {patience} epok.")
            break

    # Przywrócenie najlepszych wag modelu
    if best_model is None:
        raise RuntimeError("Trening nie zakończył się poprawnie.")

    model.load_state_dict(best_model)

    # Zapisanie najlepszych wag pod unikalną nazwą eksperymentu
    results_dir = Path(__file__).resolve().parent / "wyniki"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = results_dir / model_name
    torch.save(model.state_dict(), model_path)
    
    print(f"Zapisano wagi modelu: {model_path}")

    print(f"Najlepsza epoka: {best_epoch}")
    print(f"Najlepszy validation loss: {best_val_loss:.4f}")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    return model, history, best_epoch, best_val_loss
