// src/services.rs

use super::api::Message;
use super::error::ServiceError;
use super::AppState;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, Special},
    token::LlamaToken,
};
use std::num::NonZeroU32;

/// Formats a conversation history into a single prompt string for the Qwen model.
fn build_prompt_from_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let turn = format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role, message.content
        );
        prompt.push_str(&turn);
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Parses the model's raw output, logs any "thoughts", and returns the clean answer.
fn parse_and_log_thoughts(raw_output: String) -> String {
    match raw_output.rfind("</think>") {
        Some(end_tag_pos) => {
            let thought_block = &raw_output[..end_tag_pos + "</think>".len()];
            let answer_block = &raw_output[end_tag_pos + "</think>".len()..];

            println!("\n🤔 ----- MODEL THOUGHT ----- 🤔");
            println!("{}", thought_block.trim());
            println!("🤔 ----- END THOUGHT ----- 🤔\n");

            answer_block.trim_start().to_string()
        }
        None => raw_output, // No thought block found
    }
}

pub fn run_inference(state: &AppState, messages: Vec<Message>) -> Result<String, ServiceError> {
    const MAX_NEW_TOKENS: i32 = 4096;
    let model = &state.model;

    // 1. Format the prompt
    let prompt_str = build_prompt_from_messages(&messages);

    // 2. Build context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(4096))
        .with_n_batch(512);
    let mut ctx = model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| ServiceError::LlamaContext(e.to_string()))?;

    // 3. Tokenize and feed prompt
    let toks = model
        .str_to_token(&prompt_str, AddBos::Never)
        .map_err(|e| ServiceError::LlamaTokenize(e.to_string()))?;
    let mut batch = LlamaBatch::new(toks.len() + (MAX_NEW_TOKENS as usize), 1);
    let last_idx = toks.len() as i32 - 1;
    for (i, t) in (0_i32..).zip(toks.iter()) {
        batch.add(*t, i, &[0], i == last_idx).unwrap();
    }
    ctx.decode(&mut batch)
        .map_err(|e| ServiceError::LlamaDecode(e.to_string()))?;

    // 4. Generate response tokens
    let eos = model.token_eos();
    let mut out = String::new();
    let mut pos = batch.n_tokens();

    for _ in 0..MAX_NEW_TOKENS {
        batch.clear();
        let logits = ctx.get_logits();
        // Simple greedy sampling
        let next_id = LlamaToken::new(
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as i32)
                .unwrap(),
        );

        if next_id == eos {
            break;
        }

        let bytes = model
            .token_to_bytes(next_id, Special::Tokenize)
            .map_err(|e| ServiceError::LlamaTokenProcess(e.to_string()))?;
        out.push_str(&String::from_utf8_lossy(&bytes));

        batch.add(next_id, pos, &[0], true).unwrap();
        ctx.decode(&mut batch)
            .map_err(|e| ServiceError::LlamaDecode(e.to_string()))?;
        pos += 1;
    }

    // 5. Post-process the output
    let final_answer = parse_and_log_thoughts(out);

    Ok(final_answer)
}
