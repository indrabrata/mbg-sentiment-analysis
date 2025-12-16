import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def freeze_lower_layers(model, freeze_until: int):
    """
    Freeze encoder layers [0 .. freeze_until-1]
    """
    for layer in model.bert.encoder.layer[:freeze_until]:
        for param in layer.parameters():
            param.requires_grad = False

    logging.info("🔒 Frozen bottom %d encoder layers", freeze_until)
