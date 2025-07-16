// src/main.rs

// Declare modules
mod api;
mod error;
mod handlers;
mod services;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};
use std::sync::Arc;

// Shared application state
pub struct AppState {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize LLaMA Backend & Model
    let backend = Arc::new(LlamaBackend::init()?);
    let model_params = LlamaModelParams::default(); // .with_n_gpu_layers(35) for GPU

    let model = Arc::new(LlamaModel::load_from_file(
        &backend,
        "C:\\Users\\karls\\Documents\\Code\\java\\chat\\ai_api\\models\\Qwen3-0.6B-Q8_0.gguf",
        &model_params,
    )?);

    println!("✅ Model loaded. Server starting at http://127.0.0.1:8080");

    // 2. Create Shared App State
    let app_state = web::Data::new(AppState { backend, model });

    // 3. Start HTTP Server
    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:3000") // Your frontend
            .allowed_methods(vec!["POST"])
            .allowed_headers(vec![actix_web::http::header::CONTENT_TYPE])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .service(handlers::infer) // Register the handler from the handlers module
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;

    Ok(())
}