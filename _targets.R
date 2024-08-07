library(targets)
library(tarchetypes)
library(lubridate)

# Set target-specific options such as packages.
tar_option_set(
  packages = c(
    "tidyverse",
    "dbplyr",
    "DBI",
    "arrow",
    "janitor",
    "zeallot"
  )
)

# load all of the R scripts in the data_extraction directory
targets::tar_source("data_extraction")

# End this file with a list of target objects.
list(
  # variables ----
  # these may need to be update as required
  tar_target(data_version, "dev"),
  # take they year and turn it into a date
  tar_target(
    extract_years,
    c(2019, 2022)
  ),
  tar_target(
    start_date,
    # don't use glue as targets cannot determine the dependency
    ymd(paste0(extract_years, "-04-01"))
  ),
  tar_target(
    data_path,
    file.path("data", year(start_date))
  ),
  tar_target(
    providers_file,
    "providers.json",
    format = "file"
  ),
  tar_target(
    providers,
    providers_file |>
      jsonlite::read_json(simplifyVector = TRUE) |>
      stringr::str_split("\\|")
  ),
  tar_target(
    params,
    {
      n <- paste(collapse = "_", providers[[1]])
      dir.create(
        file.path(data_path, n),
        showWarnings = FALSE,
        recursive = TRUE
      )
      list(
        name = n,
        providers = providers[[1]],
        start_date = start_date,
        end_date = start_date %m+% lubridate::years(1) %m-% days(1),
        path = data_path
      )
    },
    pattern = cross(providers, map(start_date, data_path))
  ),
  tar_target(
    rtt_specs,
    c(
      "100",
      "101",
      "110",
      "120",
      "130",
      "140",
      "150",
      "160",
      "170",
      "300",
      "301",
      "320",
      "330",
      "340",
      "400",
      "410",
      "430",
      "502"
    )
  ),
  # targets ----
  # sql data
  tar_target(
    ip_data,
    create_provider_ip_extract(
      params,
      specialties = rtt_specs
    ),
    pattern = map(params),
    format = "file"
  ),
  tar_target(
    ip_strategies,
    create_provider_ip_strategies(
      params
    ),
    pattern = map(params),
    format = "file"
  ),
  tar_target(
    ip_synth,
    create_ip_synth(ip_data, rtt_specs),
    format = "file"
  ),
  tar_target(
    op_data,
    create_provider_op_extract(
      params,
      specialties = rtt_specs
    ),
    pattern = map(params),
    format = "file"
  ),
  tar_target(
    ecds_raw,
    "data/raw/ecds.parquet",
    format = "file"
  ),
  tar_target(
    successors_file,
    "data/reference/successors.csv",
    format = "file"
  ),
  tar_target(
    aae_data,
    generate_aae_from_ecds(
      params,
      ecds_raw,
      successors_file
    ),
    format = "file"
  ),
  # demographic factors
  tar_target(
    trust_wt_catchment_pops,
    get_trust_wt_catchment_pops()
  ),
  tar_target(
    trust_wt_catchment_births,
    get_trust_wt_catchment_births()
  ),
  tar_target(
    variant_lookup,
    get_variant_lookup()
  ),
  tar_target(
    processed_demographic_factors,
    process_demographic_factors(
      trust_wt_catchment_pops,
      variant_lookup
    )
  ),
  tar_target(
    processed_birth_factors,
    process_demographic_factors(
      trust_wt_catchment_births |>
        dplyr::mutate(sex = 2),
      variant_lookup
    )
  ),
  tar_target(
    demographic_factors,
    save_demographic_factors(
      processed_demographic_factors,
      params,
      "demographic_factors.csv"
    ),
    pattern = map(params),
    format = "file"
  ),
  tar_target(
    birth_factors,
    save_demographic_factors(
      processed_birth_factors,
      params,
      "birth_factors.csv"
    ),
    pattern = map(params),
    format = "file"
  ),
  tar_target(
    py_gam_file_path,
    "model/hsa_gams.py",
    format = "file"
  ),
  tar_target(
    gams_file_paths,
    callr::r(
      \(fn, p, b, ...) fn(p, b),
      args = list(
        fn = create_gams,
        p = params$providers,
        b = as.character(lubridate::year(params$start_date)),
        ip_data,
        op_data,
        aae_data,
        demographic_factors,
        py_gam_file_path
      )
    ),
    pattern = map(params, ip_data, op_data, demographic_factors)
  ),
  tar_target(
    gams,
    gams_file_paths,
    pattern = map(gams_file_paths),
    format = "file"
  ),
  # files upload
  tar_files(
    all_files,
    c(
      ip_data,
      ip_strategies,
      op_data,
      aae_data,
      demographic_factors,
      birth_factors,
      gams,
      # bit of a cheat, the gams only returns a single file name (of the pkl object)
      # add in the hsa_activity_table.csv files
      stringr::str_replace(gams, "_gams.pkl", "_activity_table.csv")
    )
  ),
  tar_target(
    uploaded_file,
    upload_file_to_azure(all_files, data_version),
    pattern = map(all_files)
  ),
  #
  tar_target(
    activity_counts,
    generate_activity_counts(ip_data, op_data, aae_data)
  )
)
