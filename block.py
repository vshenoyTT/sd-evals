def run_interactive_demo_inference(device, num_inference_steps, image_size=(256, 256)):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None
    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    with open(prompt_file, 'r') as file:
        prompts = file.readlines()

    for i, prompt in enumerate(prompts):
        ttnn_scheduler.set_timesteps(num_inference_steps)
        input_prompt = [prompt]

        experiment_name = f"interactive_{height}x{width}_prompt_{i}"
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        total_accum = 0
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            t0 = time.time()
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
            with torch.no_grad():
                ttnn_output = model(
                    ttnn_latent_model_input,  # input
                    timestep=_t,
                    encoder_hidden_states=ttnn_text_embeddings,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
            total_accum += time.time() - t0
            iter += 1
        print(f"Time taken for {iter} iterations: total: {total_accum:.3f}")

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)
