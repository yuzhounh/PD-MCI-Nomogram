library(rms)
library(jsonlite)

# Read the training data.
train_data <- read.csv("PPMI_9_train.csv")

# Read the feature mapping as the sole source for original names and abbreviations.
feature_mapping <- read.csv("PPMI_feature_mapping.csv")
abbr_lut <- setNames(
    as.character(feature_mapping$abbreviation), feature_mapping$feature
)

# Return names absent from the mapping (Intercept, PATNO, MCI) unchanged.
to_abbr <- function(x) unname(ifelse(is.na(abbr_lut[x]), x, abbr_lut[x]))

# Replace column names with abbreviations.
train_data_mapped <- train_data
colnames(train_data_mapped) <- to_abbr(colnames(train_data))

# Set data distributions and enforce physiological/defined limits to prevent invalid nomogram ticks, such as GDS = 16.
dd <- datadist(train_data_mapped)
if ("GDS" %in% colnames(train_data_mapped)) {
    dd$limits["High:prediction", "GDS"] <- min(
        as.numeric(dd$limits["High:prediction", "GDS"]), 15
    )
    dd$limits["High", "GDS"] <- min(as.numeric(dd$limits["High", "GDS"]), 15)
}
if ("PIGD" %in% colnames(train_data_mapped)) {
    dd$limits["High:prediction", "PIGD"] <- min(
        as.numeric(dd$limits["High:prediction", "PIGD"]), 4.0
    )
    dd$limits["High", "PIGD"] <- min(as.numeric(dd$limits["High", "PIGD"]), 4.0)
}
options(datadist = "dd")

# Read full-precision statsmodels LR coefficients from the machine-readable parameter file.
model_params <- fromJSON("model_parameters.json", simplifyVector = TRUE)
raw_coef_names <- names(model_params$coefficients)
stopifnot(length(raw_coef_names) > 0, !is.null(model_params$intercept))

# Extract all predictors dynamically and map them to abbreviations.
pred_vars <- to_abbr(raw_coef_names)
message(sprintf(
    "Found %d predictor variables in model_parameters.json: %s",
    length(pred_vars), paste(pred_vars, collapse = ", ")
))

# Build a dynamic logistic-regression formula, escaping hyphenated abbreviations with backticks.
formula_str <- paste("MCI ~", paste(sprintf("`%s`", pred_vars), collapse = " + "))
fit <- lrm(as.formula(formula_str), data = train_data_mapped, x = TRUE, y = TRUE)

# Replace lrm coefficients with the full-precision coefficients from model_parameters.json.
# Only coefficients are replaced; fit$var (covariance), fit$stats, and fit$linear.predictors
# remain from the R fit. Nomogram plotting uses only coefficients and datadist; however,
# statistics from print(fit), validate(), and calibrate() correspond to the R model and
# must not be cited as statsmodels cluster-robust estimates.
coef_raw <- c(
    Intercept = as.numeric(model_params$intercept),
    setNames(as.numeric(unlist(model_params$coefficients, use.names = FALSE)), pred_vars)
)
new_coefs <- fit$coefficients
for (nm in names(new_coefs)) {
    clean_nm <- gsub("^`|`$", "", nm)
    if (!clean_nm %in% names(coef_raw)) {
        stop(sprintf("Coefficient for '%s' not found in model_parameters.json", clean_nm))
    }
    new_coefs[nm] <- coef_raw[clean_nm]
}
fit$coefficients <- new_coefs

# Verify that all coefficients have been replaced and none are missing.
stopifnot(
    identical(length(fit$coefficients), length(names(fit$coefficients))),
    !anyNA(fit$coefficients)
)

# Create the nomogram with GDS capped at 15 and ticks spaced by 3.
nomo <- nomogram(
    fit,
    GDS = seq(0, 15, by = 3),
    fun = plogis,
    fun.at = c(0.01, 0.05, seq(0.1, 0.9, 0.1), 0.95, 0.99),
    funlabel = "Risk of PD-MCI"
)

# Save as PDF.
pdf_path <- "step_16_nomogram.pdf"

pdf(pdf_path, width = 12, height = 8)
par(lwd = 1.5)
plot(nomo, xfrac = 0.24, cex.var = 1.2, cex.axis = 1.0)
dev.off()

# Crop white margins and overwrite step_16_nomogram.pdf while preserving vector graphics.
if (Sys.which("pdfcrop") != "") {
    system2("pdfcrop", c(pdf_path, pdf_path))
}
