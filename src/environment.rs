use once_cell::sync::OnceCell;
use ort::environment::Environment;

pub(crate) fn environment() -> Result<&'static Environment, ort::Error> {
    static ENVIRONMENT: OnceCell<Environment> = OnceCell::new();
    ENVIRONMENT.get_or_try_init(|| ort::init().build())
}

#[cfg(test)]
mod tests {
    #[test]
    fn session_builders_share_an_environment() {
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    ort::session::Session::builder(super::environment().unwrap()).unwrap();
                });
            }
        });
        ort::session::Session::builder(super::environment().unwrap()).unwrap();
    }
}
