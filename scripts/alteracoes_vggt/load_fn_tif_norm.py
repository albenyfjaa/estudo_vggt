import torch
import numpy as np
import rasterio

def load_and_preprocess_geotiff(image_path_list, mode="crop", target_size=None, is_mask_or_mds=False):
    """
    Carrega e pré-processa imagens GeoTIFF.

    Args:
        image_path_list (list): Lista de caminhos para arquivos GeoTIFF.
        mode (str, optional): Modo de pré-processamento, "crop" ou "pad".
        target_size (tuple, optional): Tupla (Altura, Largura) desejada para a saída.
                                       Default é (518, 518) se não fornecido.
        is_mask_or_mds (bool, optional): True se os dados carregados são uma máscara ou MDS (para
                                         evitar normalização inadequada). Default é False.
    Returns:
        images (torch.Tensor): Tensor em batch de imagens pré-processadas (N, C, H, W).
        meta_list (list): Lista de dicionários de metadados para cada imagem.
    """
    if not image_path_list:
        raise ValueError("Pelo menos 1 imagem é necessária")
    if mode not in ["crop", "pad"]:
        raise ValueError("Modo deve ser 'crop' ou 'pad'")

    internal_scalar_target_dim = 518
    final_target_h, final_target_w = 518, 518

    if target_size is not None:
        final_target_h, final_target_w = target_size
        if final_target_h != final_target_w:
            print(f"Aviso: load_and_preprocess_geotiff recebeu um target_size não quadrado: {target_size}. "
                  f"A lógica de redimensionamento interno usará a dimensão da altura ({final_target_h}) como base escalar.")
        internal_scalar_target_dim = final_target_h
    else:
        print("Aviso: target_size não foi fornecido para load_and_preprocess_geotiff. Usando default (518, 518).")

    # Definir os tensores de média e desvio padrão para normalização (padrão ImageNet/VGG)
    # O formato .view(3, 1, 1) é para que os tensores possam ser subtraídos/divididos
    # de uma imagem com formato (C, H, W), onde C=3.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images_processed = []
    meta_list = []

    for image_path in image_path_list:
        with rasterio.open(image_path) as src:
            meta = src.meta.copy()
            image_array = src.read().astype(np.float32)
        meta_list.append(meta)

        img = torch.from_numpy(image_array)

        # Normalização condicional
        if not is_mask_or_mds:
            # Etapa 1: Escalar para [0, 1] se a imagem estiver em outra escala (ex: 0-255)
            # Esta verificação é importante pois a normalização mean/std assume o intervalo [0, 1]
            if meta.get('dtype') == 'uint8' or (img.max() > 1.0 and img.min() >= 0):
                print(f"Escalando imagem {image_path} para o intervalo [0, 1].")
                img = img / 255.0

            # Etapa 2: Aplicar normalização Z-score (mean/std)
            # Garante que os tensores de mean/std estejam no mesmo dispositivo que a imagem (CPU ou GPU)
            device = img.device
            mean_d = mean.to(device)
            std_d = std.to(device)

            if img.shape[0] == 3: # Para imagens RGB
                print(f"Aplicando normalização Mean/Std em imagem de 3 canais.")
                img = (img - mean_d) / std_d
            elif img.shape[0] == 4: # Para imagens RGBA
                print(f"Aplicando normalização Mean/Std nos 3 primeiros canais de imagem RGBA.")
                # Normaliza apenas os canais RGB e mantém o canal Alpha intacto
                img[:3, :, :] = (img[:3, :, :] - mean_d) / std_d

        c, height, width = img.shape
        inter_h, inter_w = height, width

        if mode == "pad":
            if width >= height:
                inter_w = internal_scalar_target_dim
                inter_h_float = height * (internal_scalar_target_dim / width)
            else:
                inter_h = internal_scalar_target_dim
                inter_w_float = width * (internal_scalar_target_dim / height)
            
            if final_target_h > 100:
                if width >= height:
                    inter_h = round(inter_h_float / 14) * 14
                else:
                    inter_w = round(inter_w_float / 14) * 14
            else:
                if width >= height:
                    inter_h = round(inter_h_float)
                else:
                    inter_w = round(inter_w_float)
        else:
            inter_w = internal_scalar_target_dim
            inter_h_float = height * (internal_scalar_target_dim / width)
            if final_target_h > 100:
                inter_h = round(inter_h_float / 14) * 14
            else:
                inter_h = round(inter_h_float)
        
        inter_h = max(1, int(inter_h))
        inter_w = max(1, int(inter_w))

        img_resized = img.unsqueeze(0)
        try:
            img_resized = torch.nn.functional.interpolate(
                img_resized, size=(inter_h, inter_w), mode="bicubic", align_corners=False
            )
        except Exception as e:
            print(f"Warning: Interpolação bicúbica falhou para {image_path} com shape {img.shape} e size {(inter_h, inter_w)} (erro: {e}). Usando bilinear.")
            img_resized = torch.nn.functional.interpolate(
                img_resized, size=(inter_h, inter_w), mode="bilinear", align_corners=False
            )
        img_resized = img_resized.squeeze(0)
        
        _current_c, current_h, current_w = img_resized.shape
        img_final_adjusted = img_resized

        if current_h > final_target_h:
            start_y = (current_h - final_target_h) // 2
            img_final_adjusted = img_final_adjusted[:, start_y:start_y + final_target_h, :]
        elif current_h < final_target_h:
            pad_top = (final_target_h - current_h) // 2
            pad_bottom = final_target_h - current_h - pad_top
            img_final_adjusted = torch.nn.functional.pad(img_final_adjusted, (0, 0, pad_top, pad_bottom), "constant", 0.0)

        current_w_after_h_adj = img_final_adjusted.shape[2]
        if current_w_after_h_adj > final_target_w:
            start_x = (current_w_after_h_adj - final_target_w) // 2
            img_final_adjusted = img_final_adjusted[:, :, start_x:start_x + final_target_w]
        elif current_w_after_h_adj < final_target_w:
            pad_left = (final_target_w - current_w_after_h_adj) // 2
            pad_right = final_target_w - current_w_after_h_adj - pad_left
            img_final_adjusted = torch.nn.functional.pad(img_final_adjusted, (pad_left, pad_right, 0, 0), "constant", 0.0)

        images_processed.append(img_final_adjusted)

    output_images_tensor = torch.stack(images_processed)
    return output_images_tensor, meta_list