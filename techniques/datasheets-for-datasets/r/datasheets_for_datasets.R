# Datasheets for Datasets (Reference Sec 47.220).
#
# Gebru et al 2018 CACM. Structured documentation covering motivation,
# composition, collection, preprocessing, uses, distribution, maintenance.

# Pure-R template + completeness checker
template <- list(
    motivation = list(purpose = "", creators = "", funding = ""),
    composition = list(n_instances = 0L, features = list(), target = "",
                            splits = list(), sensitive_attributes = list(),
                            recommended_use_case = ""),
    collection = list(when = "", how = "", consent_process = "",
                          ethical_review = ""),
    preprocessing = list(cleaning_steps = list(), filtering_criteria = list(),
                             known_biases = list()),
    uses = list(intended = list(), prohibited = list(),
                   known_deployments = list()),
    distribution = list(license = "", access_url = "", hosting_org = ""),
    maintenance = list(maintainer = "", contact = "",
                          update_schedule = "", erratum_process = ""))

check_completeness <- function(sheet) {
    missing <- character()
    for (sec in names(sheet)) {
        for (f in names(sheet[[sec]])) {
            v <- sheet[[sec]][[f]]
            if (identical(v, "") || identical(v, list()) || identical(v, 0L)) {
                missing <- c(missing, paste(sec, f, sep = "."))
            }
        }
    }
    missing
}

cat("=== Datasheets for Datasets (Gebru et al 2018) ===\n")
cat("Template sections: ", paste(names(template), collapse = ", "), "\n")
cat("Missing (empty template): ", length(check_completeness(template)), "\n")
cat("\nFill this template in for every dataset release. See the Python demo\n")
cat("for a fully-populated healthcare example.\n")
