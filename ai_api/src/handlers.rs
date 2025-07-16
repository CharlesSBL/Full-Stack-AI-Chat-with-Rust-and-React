// src/handlers.rs

use super::api::{InferRequest, InferResponse};
use super::error::ServiceError;
use super::services;
use super::AppState;
use actix_web::{post, web, HttpResponse, Responder};

#[post("/infer")]
pub async fn infer(
    state: web::Data<AppState>,
    body: web::Json<InferRequest>,
) -> Result<impl Responder, ServiceError> {
    // Move the inference logic to a blocking thread, as it's CPU-intensive.
    let result = web::block(move || {
        let messages = body.into_inner().messages;
        services::run_inference(&state, messages)
    })
    .await
    .map_err(|_e| ServiceError::InternalError)??; // First ? handles web::block error, second ? handles ServiceError

    Ok(HttpResponse::Ok().json(InferResponse {
        generated_text: result,
    }))
}