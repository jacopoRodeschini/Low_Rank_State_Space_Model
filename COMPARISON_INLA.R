library(dplyr)
library(ggplot2)
library(INLA)
library(sf)
library(fmesher)

# ---------------------------------------------------------
# 1. Setup & Initialization
# ---------------------------------------------------------
PATH = "..."
if(dir.exists(PATH)) setwd(PATH)

# Setup Main Results File
if (!file.exists("INLA_results.csv")) {
  results_header <- data.frame(
    id = integer(), diff_time = numeric(), ndays = integer(),
    from = as.Date(character()), to = as.Date(character()),
    RMSE_train_PM2.5 = numeric(), RMSE_train_PM10 = numeric(),
    RMSE_test_PM2.5 = numeric(), RMSE_test_PM10 = numeric(),
    time = numeric(), stringsAsFactors = FALSE
  )
  write.csv(results_header, "INLA_results.csv", row.names = FALSE)
}

# Load existing results to avoid re-running
existing_results <- read.csv("INLA_results.csv", stringsAsFactors = FALSE)
existing_results$from <- as.Date(existing_results$from)
existing_results$to   <- as.Date(existing_results$to)

set.seed(2026)
diff_times = sample.int(600, 100)[1:10]

# Pre-load Point Data
p_train_10 = read.csv(paste0(PATH, "points_train_0.csv"), header = FALSE)  
p_train_25 = read.csv(paste0(PATH, "points_train_1.csv"), header = FALSE)
p_test_10  = read.csv(paste0(PATH, "points_test_0.csv"),  header = FALSE)  
p_test_25  = read.csv(paste0(PATH, "points_test_1.csv"),  header = FALSE)

# ---------------------------------------------------------
# 2. Main Loop
# ---------------------------------------------------------

for(i in 1:length(diff_times)) {
  
  diff_time = diff_times[i]

  
  for(ndays in c(15,30,60)) {
    
    print(paste("Processing: i =", i, "| diff_time =", diff_time))
    
    # --- Data Loading ---
    df = read.csv(paste0(PATH, "output.csv")) 
    
    nms = c("time", "Latitude", "Longitude", "Altitude",
            "mean_PM2.5", "mean_PM10",
            "ERA5Land_t2m", "ERA5Land_rh", "ERA5Land_windspeed") 
    
    df = df[, 1:length(nms)]
    names(df) = nms
    
    # Use pre-loaded point data
    point_train_PM10 = p_train_10
    point_train_PM25 = p_train_25
    point_test_PM10  = p_test_10
    point_test_PM25  = p_test_25
    
    df$time = as.Date(df$time)
    last_date <- max(df$time, na.rm = TRUE)
    TO = last_date - diff_time
    FROM = TO - ndays
    
    # Check if already processed
    already_done <- any(
      existing_results$diff_time == diff_time &
        existing_results$ndays     == ndays &
        existing_results$from      == FROM &
        existing_results$to        == TO
    )
    
    if (already_done) {
      print("Skipping - Already calculated.")
      next 
    }
    
    # Filter Data by Date
    df <- df %>% dplyr::filter(time <= TO) %>% dplyr::filter(time > FROM)
    unique_times = sort(unique(df$time))
    n_times = length(unique_times)
    df$time_idx = as.integer(factor(df$time, levels = unique_times))
    
    # --- Spatial Identifiers ---
    df <- df %>%
      group_by(Longitude, Latitude) %>%
      mutate(id = cur_group_id()) %>%
      ungroup()
    
    locs <- df %>%
      group_by(id) %>%
      slice(1) %>%
      dplyr::select(Longitude, Latitude) %>%
      as.matrix() %>%
      as.data.frame()
    
    # Tag locations
    locs$train_PM10 = (locs$Longitude %in% point_train_PM10[,1]) & (locs$Latitude %in% point_train_PM10[,2])
    locs$train_PM25 = (locs$Longitude %in% point_train_PM25[,1]) & (locs$Latitude %in% point_train_PM25[,2])
    locs$test_PM10  = (locs$Longitude %in% point_test_PM10[,1])  & (locs$Latitude %in% point_test_PM10[,2])
    locs$test_PM25  = (locs$Longitude %in% point_test_PM25[,1])  & (locs$Latitude %in% point_test_PM25[,2])
    
    df_train25 = df[df$id %in% locs[locs$train_PM25, "id"],] 
    df_train10 = df[df$id %in% locs[locs$train_PM10, "id"],] 
    df_test25  = df[df$id %in% locs[locs$test_PM25,  "id"],] 
    df_test10  = df[df$id %in% locs[locs$test_PM10,  "id"],] 
    
    df_train = rbind(df_train25, df_train10) %>% unique()
    df_test  = rbind(df_test25,  df_test10)  %>% unique()
    
    df_model = df_train
    n_data = nrow(df_model)
    
    # ---------------------------------------------------------
    # Mesh & SPDE
    # ---------------------------------------------------------
    mesh25 <- inla.mesh.2d(
      loc = locs[locs$train_PM25, c("Longitude", "Latitude")],
      cutoff = 0.2,            
      max.edge = c(3, 4),       
      offset = c(0.5, 1.0)  
    )
    mesh10 <- inla.mesh.2d(
      loc = locs[locs$train_PM10, c("Longitude", "Latitude")],
      cutoff = 0.2,             
      max.edge = c(3, 4),        
      offset = c(0.5, 1.0)  
    )
    
    spde25 <- inla.spde2.pcmatern(mesh = mesh25, prior.range = c(0.5, 0.5), prior.sigma = c(1, NA)) 
    spde10 <- inla.spde2.pcmatern(mesh = mesh10, prior.range = c(0.5, 0.5), prior.sigma = c(1, NA))
    
    # ---------------------------------------------------------
    # Indices & Projection
    # ---------------------------------------------------------
    index_z1 <- inla.spde.make.index("z1", n.spde = mesh25$n, n.group = n_times)
    index_z2 <- inla.spde.make.index("z2", n.spde = mesh10$n, n.group = n_times)
    
    # Copy Indices
    index_z1_a <- inla.spde.make.index("z1_a", n.spde = mesh25$n, n.group = n_times)
    index_z2_b <- inla.spde.make.index("z2_b", n.spde = mesh10$n, n.group = n_times)
    index_z1_c <- inla.spde.make.index("z1_c", n.spde = mesh25$n, n.group = n_times)
    index_z2_d <- inla.spde.make.index("z2_d", n.spde = mesh10$n, n.group = n_times)
    
    # A Matrices
    A_est_25 <- inla.spde.make.A(mesh = mesh25, loc = as.matrix(df_model[, c("Longitude", "Latitude")]), group = df_model$time_idx, n.group = n_times)
    A_est_10 <- inla.spde.make.A(mesh = mesh10, loc = as.matrix(df_model[, c("Longitude", "Latitude")]), group = df_model$time_idx, n.group = n_times)
    
    A_test_25 <- inla.spde.make.A(mesh = mesh25, loc = as.matrix(df_test[, c("Longitude", "Latitude")]), group = df_test$time_idx, n.group = n_times)
    A_test_10 <- inla.spde.make.A(mesh = mesh10, loc = as.matrix(df_test[, c("Longitude", "Latitude")]), group = df_test$time_idx, n.group = n_times)
    
    # ---------------------------------------------------------
    # Stack Construction
    # ---------------------------------------------------------
    effects_pm25 <- list(index_z1_a, index_z2_b, list(b0_pm25=1, alt_pm25=df_model$Altitude, temp_pm25=df_model$ERA5Land_t2m, wind_pm25=df_model$ERA5Land_windspeed, rh_pm25=df_model$ERA5Land_rh))
    stack_pm25 <- inla.stack(data = list(y = cbind(df_model$mean_PM2.5, NA)), A = list(A_est_25, A_est_10, 1), effects = effects_pm25, tag = "pm25")
    
    effects_pm10 <- list(index_z1_c, index_z2_d, list(b0_pm10=1, alt_pm10=df_model$Altitude, temp_pm10=df_model$ERA5Land_t2m, wind_pm10=df_model$ERA5Land_windspeed, rh_pm10=df_model$ERA5Land_rh))
    stack_pm10 <- inla.stack(data = list(y = cbind(NA, df_model$mean_PM10)), A = list(A_est_25, A_est_10, 1), effects = effects_pm10, tag = "pm10")
    
    # Test Stacks
    n_test <- nrow(df_test)
    effects_test_25 <- list(index_z1_a, index_z2_b, list(b0_pm25=1, alt_pm25=df_test$Altitude, temp_pm25=df_test$ERA5Land_t2m, wind_pm25=df_test$ERA5Land_windspeed, rh_pm25=df_test$ERA5Land_rh))
    stack_pm25_test <- inla.stack(data = list(y = matrix(NA, n_test, 2)), A = list(A_test_25, A_test_10, 1), effects = effects_test_25, tag = "pm25_test")
    
    effects_test_10 <- list(index_z1_c, index_z2_d, list(b0_pm10=1, alt_pm10=df_test$Altitude, temp_pm10=df_test$ERA5Land_t2m, wind_pm10=df_test$ERA5Land_windspeed, rh_pm10=df_test$ERA5Land_rh))
    stack_pm10_test <- inla.stack(data = list(y = matrix(NA, n_test, 2)), A = list(A_test_25, A_test_10, 1), effects = effects_test_10, tag = "pm10_test")
    
    # Source Stacks
    stack_source_z1 <- inla.stack(data = list(y = matrix(NA, length(index_z1$z1), 2)), A = list(1), effects = list(index_z1), tag = "source_z1", remove.unused = FALSE)
    stack_source_z2 <- inla.stack(data = list(y = matrix(NA, length(index_z2$z2), 2)), A = list(1), effects = list(index_z2), tag = "source_z2", remove.unused = FALSE)
    
    full_stack <- inla.stack(stack_pm25, stack_pm10, stack_pm25_test, stack_pm10_test, stack_source_z1, stack_source_z2)
    
    # ---------------------------------------------------------
    # Run INLA
    # ---------------------------------------------------------
    formula <- y ~ -1 + 
      b0_pm25 + alt_pm25 + temp_pm25 + wind_pm25 + rh_pm25 + 
      b0_pm10 + alt_pm10 + temp_pm10 + wind_pm10 + rh_pm10 + 
      f(z1, model = spde25, group = z1.group, control.group = list(model = "ar1")) + 
      f(z2, model = spde10, group = z2.group, control.group = list(model = "ar1")) + 
      f(z1_a, copy = "z1", fixed = FALSE, group = z1_a.group) + 
      f(z2_b, copy = "z2", fixed = FALSE, group = z2_b.group) + 
      f(z1_c, copy = "z1", fixed = FALSE, group = z1_c.group) + 
      f(z2_d, copy = "z2", fixed = FALSE, group = z2_d.group)
    

    result <- tryCatch({
      inla(
        formula,
        data = inla.stack.data(full_stack),
        family = c("gaussian", "gaussian"),
        control.predictor = list(A = inla.stack.A(full_stack), compute = TRUE, link = 1),
        control.compute   = list(dic = FALSE, waic = FALSE, config = FALSE),
        control.inla = list(
          int.strategy = "eb", 
          strategy = "gaussian" 
        ),
        num.threads = 160, 
        verbose = FALSE
      )
    }, error = function(e) {
      print(paste("INLA Error:", e))
      return(NULL)
    })
    
    # ---------------------------------------------------------
    # Extract & Save
    # ---------------------------------------------------------
    if(!is.null(result)){
      
      rmse <- function(obs, pred) sqrt(mean((obs - pred)^2, na.rm = TRUE))
      fitted_all <- result$summary.fitted.values
      
      idx_train_25 <- inla.stack.index(full_stack, tag = "pm25")$data
      idx_train_10 <- inla.stack.index(full_stack, tag = "pm10")$data
      idx_test_25  <- inla.stack.index(full_stack, tag = "pm25_test")$data
      idx_test_10  <- inla.stack.index(full_stack, tag = "pm10_test")$data
      
      yhat_tr25 <- fitted_all[idx_train_25, "mean"]; yhat_tr10 <- fitted_all[idx_train_10, "mean"]
      yhat_te25 <- fitted_all[idx_test_25, "mean"];  yhat_te10 <- fitted_all[idx_test_10, "mean"]
      
      inla_time <- if (!is.null(result$cpu.used["Total"])) result$cpu.used["Total"] else NA
      
      new_row <- data.frame(
        id = i, diff_time = diff_time, ndays = ndays, from = FROM, to = TO,
        RMSE_train_PM2.5 = rmse(df_model$mean_PM2.5, yhat_tr25),
        RMSE_train_PM10  = rmse(df_model$mean_PM10, yhat_tr10),
        RMSE_test_PM2.5  = rmse(df_test$mean_PM2.5, yhat_te25),
        RMSE_test_PM10   = rmse(df_test$mean_PM10, yhat_te10),
        time = inla_time, stringsAsFactors = FALSE
      )
      
      # Append to the main file
      write.table(
        new_row,
        "INLA_results.csv",
        sep = ",",
        col.names = FALSE,
        row.names = FALSE,
        append = TRUE
      )
      
      print("Results saved.")
      
      # Cleanup Memory
      rm(result, full_stack, fitted_all, spde25, spde10, mesh25, mesh10, A_est_25, A_est_10)
      gc(verbose = FALSE)
    }
  } # end ndays
} # end i