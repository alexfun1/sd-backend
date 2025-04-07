from compel import Compel, ReturnedEmbeddingsType


def prompt_embedding(pipeline, model_type, prompt, neg_prompt):
    """
    Generate the prompt embedding using Compel.
    """
    if prompt is None:
        prompt = "1girl, long hair, breasts, smile, open mouth, blue eyes, large breasts, blonde hair, thighhighs, medium breasts, standing, nipples, underwear, ass, hetero, nude, teeth, multiple boys, penis, pussy, tongue, solo focus, indoors, looking back, dark skin, 2boys, sex, grin, white thighhighs, bra, vaginal, see-through, lips, pubic hair, uncensored, kneeling, makeup, anus, bed, no panties, bottomless, erection, chair, female pubic hair, garter straps, testicles, dark-skinned male, table, 3boys, underwear only, curtains, sex from behind, bent over, lingerie, all fours, lipstick, group sex, anal, male pubic hair, lace trim, lace, handjob, white bra, hand on another's head, doggystyle, garter belt, threesome, large penis, interracial, hanging breasts, gangbang, mmf threesome, imminent penetration, bedroom, lace-trimmed legwear, double penetration, foreskin, grabbing another's hair, netorare, spitroast, cheating (relationship), lace bra"
    if neg_prompt is None:
        neg_prompt = "bad anatomy, bad hands, bad feet, ugly, blurry, out of focus, low quality, worst quality, lowres, normal quality, jpeg artifacts, signature, watermark, username, artist name"
    if model_type.upper() == "SDXL":
        compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        conditioning, pooled = compel([prompt, neg_prompt])
        return conditioning, pooled
    elif model_type.upper() == "SD15":
        compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
        conditioning = compel(prompt)
        #return compel.build_conditioning_tensor(prompt), compel.build_conditioning_tensor(neg_prompt)
        return conditioning, None
    else:
        print(f"Unknown model type: {model_type}")
        return None