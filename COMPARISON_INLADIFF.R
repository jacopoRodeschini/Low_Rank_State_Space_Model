library(dplyr)
library(ggplot2)
library(sf)
library(sp)
library(fmesher)
library(INLA)
library(INLAspacetime)

# ---------------------------------------------------------
# 0. Settings and Path
# ---------------------------------------------------------
PATH = "..."
if(dir.exists(PATH)) setwd(PATH)

OUT_FILE = "INLADIFF_results.csv"

# ---------------------------------------------------------
# 1. Data Loading & Setup
# ---------------------------------------------------------
message("Loading base datasets...")
full_df = read.csv(paste0(PATH, "output.csv"))

nms = c("time", "Latitude", "Longitude","Altitude",
        "mean_PM2.5","mean_PM10", "ERA5Land_t2m",
        "ERA5Land_rh", "ERA5Land_windspeed")

full_df = full_df[,1:length(nms)]
names(full_df) = nms
full_df$time = as.Date(full_df$time)

point_train_PM10 = read.csv(paste0(PATH, "points_train_0.csv"), header = FALSE)  
point_train_PM25 = read.csv(paste0(PATH, "points_train_1.csv"), header = FALSE)
point_test_PM10  = read.csv(paste0(PATH, "points_test_0.csv"),  header = FALSE)  
point_test_PM25  = read.csv(paste0(PATH, "points_test_1.csv"),  header = FALSE)

# Setup Results File
if (!file.exists(OUT_FILE)) {
  results_header <- data.frame(
    id = integer(), diff_time = numeric(), ndays = integer(), model = character(),
    from = as.Date(character()), to = as.Date(character()),
    RMSE_train_PM2.5 = numeric(), RMSE_train_PM10 = numeric(),
    RMSE_test_PM2.5 = numeric(), RMSE_test_PM10 = numeric(),
    time = numeric(), stringsAsFactors = FALSE
  )
  write.csv(results_header, OUT_FILE, row.names = FALSE)
}

# Load existing results to allow skipping
existing_results <- read.csv(OUT_FILE, stringsAsFactors = FALSE)
existing_results$from <- as.Date(existing_results$from)
existing_results$to   <- as.Date(existing_results$to)

set.seed(2026)
diff_times = sample.int(600, 100)[1:10]

# ---------------------------------------------------------
# 2. Sequential Loop
# ---------------------------------------------------------
for(i in 1:length(diff_times)) {
  
  diff_time = diff_times[i]
  
  for(ndays in c(15,30,60)) {
    
    print(paste("Processing: i =", i, "| diff_time =", diff_time, "| ndays =", ndays))
    
    # ---------------------------------------------------------
    # A. Data Filtering
    # ---------------------------------------------------------
    last_date <- max(full_df$time, na.rm = TRUE)
    TO = last_date - diff_time
    FROM = TO - ndays
    
    # Check if already processed
    already_done <- any(
      existing_results$diff_time == diff_time &
        existing_results$ndays     == ndays &
        existing_results$from      == FROM &
        existing_results$to        == TO &
        existing_results$model     == "102_LCM"
    )
    
    if (already_done) {
      print("Skipping - Already calculated.")
      next 
    }
    
    df <- full_df %>% dplyr::filter(time <= TO) %>% dplyr::filter(time > FROM)
    unique_times = sort(unique(df$time))
    df$time_idx = as.integer(factor(df$time, levels = unique_times))
    n_times <- length(unique_times)
    
    # Locations
    df <- df %>% group_by(Longitude, Latitude) %>% mutate(id = cur_group_id()) %>% ungroup()
    
    locs <- df %>% group_by(id) %>% slice(1) %>% dplyr::select(Longitude, Latitude) %>% as.matrix() %>% as.data.frame()
    locs$train_PM10 = (locs$Longitude %in% point_train_PM10[,1]) & (locs$Latitude %in% point_train_PM10[,2])
    locs$train_PM25 = (locs$Longitude %in% point_train_PM25[,1]) & (locs$Latitude %in% point_train_PM25[,2])
    locs$test_PM10  = (locs$Longitude %in% point_test_PM10[,1])  & (locs$Latitude %in% point_test_PM10[,2])
    locs$test_PM25  = (locs$Longitude %in% point_test_PM25[,1])  & (locs$Latitude %in% point_test_PM25[,2])
    
    df_run = df[df$id %in% locs[locs$train_PM10 | locs$train_PM25, "id"],] 
    df_test  = df[df$id %in% locs[locs$test_PM10  | locs$test_PM25,  "id"],] 
    
    # ---------------------------------------------------------
    # B. Dual Meshes & 1D Time Mesh
    # ---------------------------------------------------------
    smesh25 <- inla.mesh.2d(
      loc = locs[locs$train_PM25, c("Longitude", "Latitude")],
      cutoff = 0.2, max.edge = c(3, 4), offset = c(0.5, 1.0)
    )
    smesh10 <- inla.mesh.2d(
      loc = locs[locs$train_PM10, c("Longitude", "Latitude")],
      cutoff = 0.2, max.edge = c(3, 4), offset = c(0.5, 1.0)
    )
    
    tmesh <- fm_mesh_1d(
      loc = seq(min(df_run$time_idx), max(df_run$time_idx), by = 1),
      degree = 1
    )
    
    # ---------------------------------------------------------
    # C. Model Definition (Diffusive 102)
    # ---------------------------------------------------------
    stmodel25 <- stModel.define(smesh = smesh25, tmesh = tmesh, model = "102",
                                control.priors = list(prs=c(0.5,0.5), prt=c(5,0.5), psigma=c(1, 0.01)))
    
    stmodel10 <- stModel.define(smesh = smesh10, tmesh = tmesh, model = "102",
                                control.priors = list(prs=c(0.5,0.5), prt=c(5,0.5), psigma=c(1, 0.01)))
    
    n_st25 <- smesh25$n * tmesh$n
    n_st10 <- smesh10$n * tmesh$n
    
    # ---------------------------------------------------------
    # D. Indices & Projection Matrices
    # ---------------------------------------------------------
    # A-matrices for Data (PM2.5 and PM10 training)
    A_25_est <- inla.spde.make.A(mesh=smesh25, loc=as.matrix(df_run[, c("Longitude","Latitude")]), group=df_run$time_idx, group.mesh=tmesh)
    A_10_est <- inla.spde.make.A(mesh=smesh10, loc=as.matrix(df_run[, c("Longitude","Latitude")]), group=df_run$time_idx, group.mesh=tmesh)
    
    # A-matrices for Test
    A_25_tst <- inla.spde.make.A(mesh=smesh25, loc=as.matrix(df_test[, c("Longitude","Latitude")]), group=df_test$time_idx, group.mesh=tmesh)
    A_10_tst <- inla.spde.make.A(mesh=smesh10, loc=as.matrix(df_test[, c("Longitude","Latitude")]), group=df_test$time_idx, group.mesh=tmesh)
    
    # ---------------------------------------------------------
    # E. Stack Construction (Full LCM)
    # ---------------------------------------------------------
    # 1. Stack PM2.5 (Training)
    stack_pm25 <- inla.stack(
      data = list(y = cbind(df_run$mean_PM2.5, NA)),
      A = list(A_25_est, A_10_est, 1), 
      effects = list(
        z1_copy1 = 1:n_st25, 
        z2_copy1 = 1:n_st10,
        list(b0_pm25=1, alt_pm25=df_run$Altitude, temp_pm25=df_run$ERA5Land_t2m, 
             wind_pm25=df_run$ERA5Land_windspeed, rh_pm25=df_run$ERA5Land_rh)
      ),
      tag = "pm25"
    )
    
    # 2. Stack PM10 (Training)
    stack_pm10 <- inla.stack(
      data = list(y = cbind(NA, df_run$mean_PM10)),
      A = list(A_25_est, A_10_est, 1),
      effects = list(
        z1_copy2 = 1:n_st25, 
        z2_copy2 = 1:n_st10,
        list(b0_pm10=1, alt_pm10=df_run$Altitude, temp_pm10=df_run$ERA5Land_t2m, 
             wind_pm10=df_run$ERA5Land_windspeed, rh_pm10=df_run$ERA5Land_rh)
      ),
      tag = "pm10"
    )
    
    # 3. Test Stacks
    stack_pm25_test <- inla.stack(
      data = list(y = matrix(NA, nrow=nrow(df_test), ncol=2)),
      A = list(A_25_tst, A_10_tst, 1),
      effects = list(
        z1_copy1 = 1:n_st25, z2_copy1 = 1:n_st10, 
        list(b0_pm25=1, alt_pm25=df_test$Altitude, temp_pm25=df_test$ERA5Land_t2m,
             wind_pm25=df_test$ERA5Land_windspeed, rh_pm25=df_test$ERA5Land_rh)
      ),
      tag = "pm25_test"
    )
    
    stack_pm10_test <- inla.stack(
      data = list(y = matrix(NA, nrow=nrow(df_test), ncol=2)),
      A = list(A_25_tst, A_10_tst, 1),
      effects = list(
        z1_copy2 = 1:n_st25, z2_copy2 = 1:n_st10, 
        list(b0_pm10=1, alt_pm10=df_test$Altitude, temp_pm10=df_test$ERA5Land_t2m,
             wind_pm10=df_test$ERA5Land_windspeed, rh_pm10=df_test$ERA5Land_rh)
      ),
      tag = "pm10_test"
    )
    
    # 4. Master Source Stack
    stack_source <- inla.stack(
      data = list(y = matrix(NA, n_st25 + n_st10, 2)),
      A = list(1, 1), 
      effects = list(
        z1_M = c(1:n_st25, rep(NA, n_st10)), 
        z2_M = c(rep(NA, n_st25), 1:n_st10)
      ),
      tag = "source",
      remove.unused = FALSE
    )
    
    full_stack <- inla.stack(stack_pm25, stack_pm10, stack_pm25_test, stack_pm10_test, stack_source)
    
    # ---------------------------------------------------------
    # F. Formula & INLA Run
    # ---------------------------------------------------------
    start_time <- Sys.time()
    
    formula <- y ~ -1 +
      b0_pm25 + alt_pm25 + temp_pm25 + wind_pm25 + rh_pm25 +
      b0_pm10 + alt_pm10 + temp_pm10 + wind_pm10 + rh_pm10 +
      # MASTER FIELDS
      f(z1_M, model = stmodel25) +
      f(z2_M, model = stmodel10) +
      # COPIES
      f(z1_copy1, copy = "z1_M", range=c(0, Inf), hyper = list(beta = list(fixed = FALSE))) + 
      f(z2_copy1, copy = "z2_M", range=c(0, Inf), hyper = list(beta = list(fixed = FALSE))) + 
      f(z1_copy2, copy = "z1_M", range=c(0, Inf), hyper = list(beta = list(fixed = FALSE))) + 
      f(z2_copy2, copy = "z2_M", range=c(0, Inf), hyper = list(beta = list(fixed = FALSE)))    
    
    result <- tryCatch({
      inla(formula, 
           data = inla.stack.data(full_stack), 
           family = c("gaussian", "gaussian"),
           control.predictor = list(A = inla.stack.A(full_stack), compute = TRUE),
           control.compute   = list(dic = FALSE, waic = FALSE, config = TRUE),
           control.inla = list(int.strategy = "eb"),
           num.threads = 160,
           verbose = FALSE) # Set to TRUE for monitoring sequential run
    }, error = function(e) {
      print(paste("INLA Error:", e))
      return(NULL)
    })
    
    if(!is.null(result)) {
      
      idx_train_pm25 <- inla.stack.index(full_stack, tag = "pm25")$data
      idx_train_pm10 <- inla.stack.index(full_stack, tag = "pm10")$data
      idx_test_pm25  <- inla.stack.index(full_stack, tag = "pm25_test")$data
      idx_test_pm10  <- inla.stack.index(full_stack, tag = "pm10_test")$data
      
      fitted_mean <- result$summary.fitted.values$mean
      
      rmse <- function(obs, pred) sqrt(mean((obs - pred)^2, na.rm = TRUE))
      
      inla_time_sec <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
      
      new_row <- data.frame(
        id = i, diff_time = diff_time, ndays = ndays, model = "102_LCM",
        from = FROM, to = TO,
        RMSE_train_PM2.5 = rmse(df_run$mean_PM2.5, fitted_mean[idx_train_pm25]),
        RMSE_train_PM10  = rmse(df_run$mean_PM10, fitted_mean[idx_train_pm10]),
        RMSE_test_PM2.5  = rmse(df_test$mean_PM2.5, fitted_mean[idx_test_pm25]),
        RMSE_test_PM10   = rmse(df_test$mean_PM10, fitted_mean[idx_test_pm10]),
        time = inla_time_sec, stringsAsFactors = FALSE
      )
      
      # Append immediately
      write.table(new_row, OUT_FILE, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
      print("Results saved.")
      
      # Cleanup memory
      rm(result, full_stack, fitted_mean, smesh25, smesh10, tmesh)
      rm(A_25_est, A_10_est, A_25_tst, A_10_tst)
      gc(verbose = FALSE)
    }
  } # End Ndays
} # End i Loop