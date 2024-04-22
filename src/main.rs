use std::sync::OnceLock as OnceCell;

use actix_cors::Cors;
use actix_multipart;
use actix_multipart::{form::{
    MultipartForm,
    tempfile::{TempFile,
               TempFileConfig}}};
use actix_web::{App, HttpServer, middleware, post, Responder};
use serde::Serialize;
use whisper_rs::FullParams;

use crate::transcriber::Transcriber;

mod audio_parser;
mod model_handler;
mod transcriber;

#[derive(Debug, MultipartForm)]
struct UploadForm {
    #[multipart(rename = "file")]
    files: Vec<TempFile>,
}

#[derive(Serialize)]
struct WhisperObj {
    start: i64,
    end: i64,
    text: String,
}

#[derive(Debug)]
struct ModelData {
    trans: Transcriber,
}

static MODEL: OnceCell<ModelData> = OnceCell::new();

async fn get_params() -> FullParams<'static, 'static> {
    let mut _params = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    _params.set_initial_prompt("以下是普通话的句子，这是一段会议记录。");
    _params.set_language(Some("zh"));
    _params.set_debug_mode(false);
    _params.set_print_progress(false);
    _params.set_print_special(false);
    _params.set_print_realtime(false);
    _params.set_print_timestamps(false);
    return _params;
}

async fn initial_model() {
    let data = ModelData {
        trans: Transcriber::new(model_handler::ModelHandler::new("large-v3-q5_0", "./model").await)
    };
    MODEL.set(data).expect("sel model error");
}


#[post("/transcribe")]
async fn transcribe(
    MultipartForm(form): MultipartForm<UploadForm>
) -> Result<impl Responder, actix_web::Error> {
    let mut texts = vec![];
    for f in form.files {
        let d = MODEL.get().unwrap();
        let params = get_params().await;
        let result = d.trans.transcribe_(Box::new(f.file.into_file()), Some(params)).unwrap();
        let text = result.get_text();
        let start = result.get_start_timestamp();
        let end = result.get_end_timestamp();
        let r = WhisperObj {
            start: *start,
            end: *end,
            text: text.to_owned(),
        };
        texts.push(r);
    }
    Ok(actix_web::HttpResponse::Ok().json(texts))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    initial_model().await;
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::default().allow_any_header().allow_any_method().allow_any_origin())
            .wrap(middleware::Logger::default())
            .app_data(TempFileConfig::default().directory("./tmp"))
            // .app_data(web::Data::new(data))
            .service(transcribe)
    })
        .bind(("127.0.0.1", 8080))?
        .workers(2)
        .run()
        .await
}
