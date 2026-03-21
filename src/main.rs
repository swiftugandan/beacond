use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    beacon::cli::execute().await
}
