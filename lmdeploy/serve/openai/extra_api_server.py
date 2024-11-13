from http import HTTPStatus

from fastapi import Depends, Request
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.api_server import (VariableInterface, check_api_key,
                                              check_request,
                                              create_error_response, logger,
                                              logit_bias_logits_processor,
                                              router)
from lmdeploy.serve.openai.extra_protocol import BatchChatCompletionRequest


@router.post('/v1/chat/batch_completions',
             dependencies=[Depends(check_api_key)])
async def batch_chat_completions_v1(request: BatchChatCompletionRequest,
                                    raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format. Chat history
        example: `[[{"role": "user", "content": "hi"}]]`.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int | None): output token nums. Default to None.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - response_format (Dict | None): Only pytorch backend support formatting
        response. Examples: `{"type": "json_schema", "json_schema": {"name":
        "test","schema": {"properties": {"name": {"type": "string"}},
        "required": ["name"], "type": "object"}}}`
        or `{"type": "regex_schema", "regex_schema": "call me [A-Za-z]{1,10}"}`
    - logit_bias (Dict): Bias to logits. Only supported in pytorch engine.
    - tools (List): A list of tools the model may call. Currently, only
        internlm2 functions are supported as a tool. Use this to specify a
        list of functions for which the model can generate JSON inputs.
    - tool_choice (str | object): Controls which (if any) tool is called by
        the model. `none` means the model will not call any tool and instead
        generates a message. Specifying a particular tool via {"type":
        "function", "function": {"name": "my_function"}} forces the model to
        call that tool. `auto` or `required` will put all the tools information
        to the model.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - min_new_tokens (int): To generate at least numbers of tokens.
    - min_p (float): Minimum token probability, which will be scaled by the
        probability of the most likely token. It must be a value between
        0 and 1. Typical values are in the 0.01-0.2 range, comparably
        selective as setting `top_p` in the 0.99-0.8 range (use the
        opposite of normal `top_p` values)

    Currently we do not support the following features:
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    adapter_name = None
    if model_name != VariableInterface.async_engine.model_name:
        adapter_name = model_name  # got a adapter name
    # created_time = int(time.time())

    if isinstance(request.stop, str):
        request.stop = [request.stop]

    gen_logprobs, logits_processors = None, None
    if request.logprobs and request.top_logprobs:
        gen_logprobs = request.top_logprobs
    response_format = None
    if request.response_format and request.response_format.type != 'text':
        if VariableInterface.async_engine.backend != 'pytorch':
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'only pytorch backend can use response_format now')
        response_format = request.response_format.model_dump()

    if request.logit_bias is not None:
        try:
            logits_processors = [
                logit_bias_logits_processor(
                    request.logit_bias,
                    VariableInterface.async_engine.tokenizer.model)
            ]
        except Exception as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    random_seed = request.seed if request.seed else None

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=True,
        logprobs=gen_logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        response_format=response_format,
        logits_processors=logits_processors,
        min_new_tokens=request.min_new_tokens,
        min_p=request.min_p,
        random_seed=random_seed)

    tools = None
    if request.tools and request.tool_choice != 'none':
        gen_config.skip_special_tokens = False
        if request.stream is True:
            logger.warning('Set stream to False for tools')
            request.stream = False
        # internlm2 only uses contents inside function regardless of 'type'
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump() for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump() for item in request.tools]

    resp = VariableInterface.async_engine.extra_batch_infer(
        prompts=request.messages,
        gen_config=gen_config,
        do_preprocess=True,
        adapter_name=adapter_name,
        use_tqdm=False)

    logger.info(f'Batch chat completion tool: {tools}')
    logger.info(f'Batch chat completion response: {resp}')

    return resp
