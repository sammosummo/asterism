use asterism::{
    LatentMediationEvaluation, LatentMediationFamilyEvaluation, LatentMediationFamilyInput,
    LatentMediationFit, LatentMediationModel, LatentMediationParameters,
};

#[test]
fn latent_mediation_rust_interface_is_public() {
    let _: Option<LatentMediationParameters> = None;
    let _: Option<LatentMediationFamilyInput> = None;
    let _: Option<LatentMediationFamilyEvaluation> = None;
    let _: Option<LatentMediationEvaluation> = None;
    let _: Option<LatentMediationFit> = None;
    let _: Option<LatentMediationModel> = None;

    let _fit: fn(&LatentMediationModel) -> Result<LatentMediationFit, &'static str> =
        LatentMediationModel::fit;
}
