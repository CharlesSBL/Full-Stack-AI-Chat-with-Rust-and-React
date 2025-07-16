// src/error.rs

use actix_web::{http::StatusCode, HttpResponse, ResponseError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("Failed to create LLaMA context: {0}")]
    LlamaContext(String),

    #[error("Failed to tokenize prompt: {0}")]
    LlamaTokenize(String),

    #[error("Failed to decode prompt: {0}")]
    LlamaDecode(String),

    #[error("Failed to process token: {0}")]
    LlamaTokenProcess(String),

    #[error("Internal Server Error")]
    InternalError,
}

// Allow Actix to convert our custom error into an HTTP response
impl ResponseError for ServiceError {
    fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code()).json(serde_json::json!({
            "error": self.to_string()
        }))
    }
}
