use actix_cors::Cors;
use actix_web::{post, web, App, HttpResponse, HttpServer, error};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, sync::Arc};

/* ---------- JSON Structures for Chat API ---------- */

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

// Implement Display to easily convert Role enum to a lowercase string
impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}


#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Deserialize)]
struct InferReq {
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct InferResp {
    generated_text: String,
}

/* ---------- Shared State ---------- */
struct AppState {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}

/// Formats a conversation history into a single prompt string for the Qwen model.
fn build_prompt_from_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();

    for message in messages {
        // Append "<|im_start|>role\ncontent<|im_end|>\n" for each message
        let turn = format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role,
            message.content
        );
        prompt.push_str(&turn);
    }

    // Add the generation prompt for the assistant
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}


/* ---------- POST /infer (Chat Endpoint) ---------- */
#[post("/infer")]
async fn infer(
    data: web::Data<AppState>,
    body: web::Json<InferReq>,
) -> actix_web::Result<HttpResponse> {
    // Clone the messages to move them into the blocking thread
    let messages = body.messages.clone();

    let text = tokio::task::spawn_blocking(move || {
        let backend = &data.backend;
        let model = &data.model;
        
        // Increased token limit to prevent truncated responses.
        const MAX_NEW_TOKENS: i32 = 4096;

        // 1. Format the conversation history into a single string
        let prompt_str = build_prompt_from_messages(&messages);

        // 2. Build context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096))
            .with_n_batch(512);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {e:?}"))?;

        // 3. Tokenize and feed the prompt into the model
        let toks = model
            .str_to_token(&prompt_str, AddBos::Never)
            .map_err(|e| format!("Failed to tokenize prompt: {e:?}"))?;
        let mut batch = LlamaBatch::new(toks.len() + (MAX_NEW_TOKENS as usize), 1);
        let last_idx = toks.len() as i32 - 1;
        for (i, t) in (0_i32..).zip(toks.iter()) {
            batch.add(*t, i, &[0], i == last_idx).unwrap();
        }
        ctx.decode(&mut batch).map_err(|e| format!("Failed to decode prompt: {e:?}"))?;

        // 4. Generate response tokens
        let eos = model.token_eos();
        let mut out = String::new();
        let mut pos = batch.n_tokens();

        for _ in 0..MAX_NEW_TOKENS {
            batch.clear();
            
            let logits = ctx.get_logits();
            let next_id = LlamaToken::new(
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as i32)
                    .unwrap(),
            );
            
            if next_id == eos { break; }

            let bytes = model.token_to_bytes(next_id, Special::Tokenize)
                .map_err(|e| format!("Failed to decode token: {e:?}"))?;
            out.push_str(&String::from_utf8_lossy(&bytes));

            batch.add(next_id, pos, &[0], true).unwrap();
            ctx.decode(&mut batch).map_err(|e| format!("Failed to decode next token: {e:?}"))?;
            pos += 1;
        }

        // --- NEW: Parse output, log thoughts, and clean the answer ---
        let final_answer = match out.rfind("</think>") {
            Some(end_tag_pos) => {
                // We found a thought block.
                // The part including the tag is the thought.
                let thought_block = &out[..end_tag_pos + "</think>".len()];
                
                // The part after the tag is the final answer for the user.
                let answer_block = &out[end_tag_pos + "</think>".len()..];

                // Log the thought process to the server's console for debugging.
                println!("\n🤔 ----- MODEL THOUGHT ----- 🤔");
                println!("{}", thought_block.trim());
                println!("🤔 ----- END THOUGHT ----- 🤔\n");
                
                // Return the cleaned answer, removing any leading whitespace.
                answer_block.trim_start().to_string()
            }
            None => {
                // No think block was found, so the entire output is the answer.
                out
            }
        };

        Ok::<String, String>(final_answer)
    })
    .await
    .unwrap() // Propagate panics
    .map_err(error::ErrorInternalServerError)?; // Map String error to 500

    Ok(HttpResponse::Ok().json(InferResp { generated_text: text }))
}
/* ---------- main ---------- */
#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = LlamaBackend::init()?;
    let backend = Arc::new(backend);

    // For better performance, consider offloading layers to the GPU
    // let model_params = LlamaModelParams::default().with_n_gpu_layers(35);
    let model_params = LlamaModelParams::default();

    let model = LlamaModel::load_from_file(
        &backend,
        "C:\\Users\\karls\\Documents\\Code\\java\\chat\\ai_api\\models\\Qwen3-0.6B-Q8_0.gguf",
        &model_params,
    )?;
    let model = Arc::new(model);

    println!("✅ Model loaded. Server starting at http://127.0.0.1:8080");

    let app_state = web::Data::new(AppState { backend, model });

    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000") // Your frontend
            .allowed_methods(vec!["POST"])
            .allowed_headers(vec![
                actix_web::http::header::CONTENT_TYPE,
            ])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .service(infer)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;
    Ok(())
}