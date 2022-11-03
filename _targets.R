library(targets)
library(tarchetypes)

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
purrr::walk(fs::dir_ls("data_extraction", glob = "*.R"), source)

# End this file with a list of target objects.
list(
  tar_target(rtt_specs, c(
    "100", "101", "110", "120", "130", "140", "160", "300", "320", "330", "400",
    "410", "430", "502"
  )),
  tar_target(providers, list(
    "RA9",
    "RD8",
    "RGP",
    "RGR",
    "RH5", # "RBA" is merged in with this activity
    "RH8", # was "RBZ",
    "RHW",
    "RN5",
    "RNQ",
    "RX1",
    "RXC",
    c("RXN", "RTX"),
    "RYJ"
  )),
  # sql data
  tar_target(
    reference_data,
    extract_reference_data("../nhp_documentation/data")
  ),
  tar_target(
    ip_data_file_paths,
    create_provider_ip_extract(providers[[1]], specialties = rtt_specs),
    pattern = map(providers)
  ),
  tar_target(
    ip_data,
    ip_data_file_paths[[1]][[1]],
    pattern = map(ip_data_file_paths),
    format = "file"
  ),
  tar_target(
    ip_files,
    ip_data_file_paths[[1]],
    pattern = map(ip_data_file_paths),
    format = "file"
  ),
  tar_target(
    ip_synth_data_file_paths,
    create_synthetic_ip_extract(specialties = rtt_specs),
  ),
  tar_target(
    ip_synth_data,
    ip_synth_data_file_paths[[1]],
    format = "file"
  ),
  tar_target(
    op_data_file_paths,
    create_provider_op_extract(providers[[1]], specialties = rtt_specs),
    pattern = map(providers)
  ),
  tar_target(
    op_data,
    op_data_file_paths,
    pattern = map(op_data_file_paths),
    format = "file"
  ),
  tar_target(
    op_synth_data,
    create_synthetic_op_extract(specialties = rtt_specs),
    format = "file"
  ),
  tar_target(
    aae_data_file_paths,
    create_provider_aae_extract(providers[[1]]),
    pattern = map(providers)
  ),
  tar_target(
    aae_data,
    aae_data_file_paths,
    pattern = map(aae_data_file_paths),
    format = "file"
  ),
  tar_target(
    aae_synth_data,
    create_synthetic_aae_extract(),
    format = "file"
  ),
  # kh03 data
  tar_target(
    kh03_base_url,
    "https://www.england.nhs.uk/statistics/statistical-work-areas/bed-availability-and-occupancy"
  ),
  tar_target(
    kh03_files_overnight,
    kh03_get_files(paste(kh03_base_url, "bed-data-overnight", sep = "/"))
  ),
  tar_target(
    kh03_files_dayonly,
    kh03_get_files(paste(kh03_base_url, "bed-data-day-only", sep = "/"))
  ),
  tar_target(
    kh03_overnight,
    kh03_get_file(kh03_files_overnight[[1]]),
    pattern = map(kh03_files_overnight)
  ),
  tar_target(
    kh03_dayonly,
    kh03_get_file(kh03_files_dayonly[[1]]) |>
      dplyr::select("quarter", "org_code", "specialty_group", available_dayonly = "available_total"),
    pattern = map(kh03_files_dayonly)
  ),
  tar_target(kh03_all, kh03_combine(kh03_overnight, kh03_dayonly)),
  tar_target(kh03_processed, kh03_process(kh03_all)),
  tar_target(
    kh03_synthetic,
    kh03_generate_synthnetic(kh03_processed),
    format = "file"
  ),
  tar_target(
    kh03_save_file_paths,
    kh03_save_trust(kh03_processed, providers[[1]]),
    pattern = map(providers)
  ),
  tar_target(
    kh03_save,
    kh03_save_file_paths,
    pattern = map(kh03_save_file_paths),
    format = "file"
  ),
  # demographic factors
  tar_target(demographic_raw_data_path, "_scratch/demographic_factors.rds", format = "file"),
  tar_target(processed_demographic_factors, process_demographic_factors(demographic_raw_data_path)),
  tar_target(
    demographic_factors_synthetic,
    save_synthetic_demographic_factors(processed_demographic_factors),
    format = "file"
  ),
  tar_target(
    demographic_factors_file_paths,
    save_demographic_factors(processed_demographic_factors, providers[[1]]),
    pattern = map(providers)
  ),
  tar_target(
    demographic_factors,
    demographic_factors_file_paths,
    pattern = map(demographic_factors_file_paths),
    format = "file"
  ),
  tar_target(
    gams_file_paths,
    callr::r(
      \(fn, p, ...) fn(p),
      args = list(
        fn = create_gams,
        p = providers[[1]],
        ip_data,
        op_data,
        aae_data,
        demographic_factors
      )
    ),
    pattern = map(providers, ip_data, op_data, aae_data, demographic_factors)
  ),
  tar_target(
    gams,
    gams_file_paths,
    pattern = map(gams_file_paths),
    format = "file"
  ),
  tar_target(
    gams_synthetic,
    callr::r(
      \(fn, p, ...) fn(p),
      args = list(
        fn = create_gams,
        p = "synthetic",
        ip_synth_data,
        op_synth_data,
        aae_synth_data,
        demographic_factors_synthetic
      )
    ),
    format = "file"
  ),
  # theatres
  tar_target(theatres_data_path, "_scratch/theatres_data.csv", format = "file"),
  tar_target(qmco_data_path, "_scratch/qmco.xlsx", format = "file"),
  tar_target(
    theatres_four_hour_sessions,
    theatres_get_four_hour_sessions(theatres_data_path)
  ),
  tar_target(
    theatres_available,
    theatres_get_available(qmco_data_path)
  ),
  tar_target(
    theatres_file_paths,
    theatres_save_data(theatres_four_hour_sessions, theatres_available, providers[[1]]),
    pattern = map(providers)
  ),
  tar_target(
    theatres,
    theatres_file_paths,
    pattern = map(theatres_file_paths),
    format = "file"
  ),
  tar_target(
    theatres_synthetic,
    theatres_generate_synthetic(theatres_four_hour_sessions, theatres_available),
    format = "file"
  ),
  # files upload
  tar_files(
    all_files,
    c(
      ip_files,
      ip_synth_data,
      op_data,
      op_synth_data,
      aae_data,
      aae_synth_data,
      demographic_factors,
      demographic_factors_synthetic,
      gams,
      gams_synthetic,
      kh03_save,
      kh03_synthetic,
      theatres,
      theatres_synthetic
    )
  ),
  tar_target(uploaded_file, upload_file_to_azure(all_files), pattern = map(all_files))
)
