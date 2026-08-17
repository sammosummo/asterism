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

    // Named, not bound to `_`: the point is that the signature type-checks, and
    // a `_` binding drops the value before the annotation has been made to
    // carry any weight.
    let fit_signature: fn(&LatentMediationModel) -> Result<LatentMediationFit, &'static str> =
        LatentMediationModel::fit;
    assert!(std::ptr::fn_addr_eq(
        fit_signature,
        LatentMediationModel::fit
            as fn(&LatentMediationModel) -> Result<LatentMediationFit, &'static str>
    ));
}
