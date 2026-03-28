use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tokio::net::TcpListener;

use crate::rest::{RestState, build_router};

pub async fn run_server(port: u16, models_root: PathBuf) -> Result<()> {
    let bind_address = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(bind_address)
        .await
        .with_context(|| format!("failed to bind REST server to `{bind_address}`"))?;

    println!();
    println!("REST server");
    println!("  listening on : http://{bind_address}");
    println!("  models root  : {}", models_root.display());
    println!();

    let state = RestState { models_root };
    axum::serve(listener, build_router(state))
        .await
        .context("REST server stopped unexpectedly")
}
