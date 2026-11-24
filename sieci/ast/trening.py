import torch
from pathlib import Path


# Wybór urządzenia obliczeniowego
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


# Trening modelu AST wraz z walidacją i early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, 
                patience=10, model_name="model_weights.pth"):
    if epochs < 1:
        raise ValueError("Liczba epok musi być większa od zera.")

    if patience < 1:
        raise ValueError("Patience musi być większe od zera.")

    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Pętla treningowa.
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * X.size(0)

        train_loss = train_loss.item() / len(train_loader.dataset)

        # Walidacja modelu.
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.detach() * X.size(0)

        val_loss = val_loss.item() / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoka {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        # Zapisanie najlepszych wag
        if val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss

            best_model = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

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

    # Zapisanie wag w folderze wyników
    results_dir = Path(__file__).resolve().parent / "wyniki"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        results_dir / model_name
    )

    print(f"Zapisano wagi modelu: {results_dir / model_name}")
    print(f"Najlepsza epoka: {best_epoch}")
    print(f"Najlepszy validation loss: {best_val_loss:.4f}")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    return model, history, best_epoch, best_val_loss
