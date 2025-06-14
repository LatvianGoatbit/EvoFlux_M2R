functions {
    vector calculate_mixing_weights(real mu, real gamma, real nu, real zeta, real tau) {
        matrix[3,3] RateMatrix;
        vector[3] mkw;
        matrix[3, 1] matrix_mkw;
        matrix[3,1] Matrix_ProbStates;
        vector[3] ProbStates;

        RateMatrix[,1] = [-2*gamma, nu, 0]';
        RateMatrix[,2] = [2*gamma, -(nu+zeta), 2*mu]';
        RateMatrix[,3] = [0, zeta, -2*mu]';

        mkw = [0.5, 0.0, 0.5]';
        matrix_mkw = to_matrix(mkw);
        Matrix_ProbStates = scale_matrix_exp_multiply(tau, RateMatrix, matrix_mkw);
        ProbStates = to_vector(Matrix_ProbStates);

        return log(ProbStates);
    }

    vector calculate_frechet_parameters(real mut_rate, real theta, real deltaT) {
        vector[3] frechet_params;
        real a = exp(1)/2;

        frechet_params[1] = exp(1)/2;
        frechet_params[2] = 0.5 * mut_rate/theta * (theta * deltaT + log(mut_rate/theta) - (1+a));
        frechet_params[3] = 0.5 * exp(1) * mut_rate / theta;

        return frechet_params;
    }
    vector homo_demeth_logpdf(vector y, real theta, real mu, real tau, int T, int N) {
        vector[3] frechet_parameters = calculate_frechet_parameters(2*mu, theta, T - tau);
        real a = frechet_parameters[1];
        real m_norm = frechet_parameters[2];
        real s_norm = frechet_parameters[3];
        vector[N] y_shifted_demeth = y - m_norm;
    
        vector[N] frechet_lpdf_demeth = rep_vector(0, N);
        for (n in 1:N) {
            if (y_shifted_demeth[n] <= 0) {
                frechet_lpdf_demeth[n] = negative_infinity();
                } else if (y_shifted_demeth[n] > 0) {
                frechet_lpdf_demeth[n] = frechet_lpdf(y_shifted_demeth[n] | a, s_norm);
                }
            }
        return frechet_lpdf_demeth;
    }

    vector homo_meth_logpdf(vector y, real theta, real gamma, real tau, int T, int N) {
        vector[3] frechet_parameters = calculate_frechet_parameters(2*gamma, theta, T - tau);
        real a = frechet_parameters[1];
        real m_norm = frechet_parameters[2];
        real s_norm = frechet_parameters[3];
        vector[N] y_shifted_meth = 1 - y - m_norm;
        
        vector[N] frechet_lpdf_meth = rep_vector(0, N);
        for (n in 1:N) {
            if (y_shifted_meth[n] <= 0) {
                frechet_lpdf_meth[n] = negative_infinity();
            } else if (y_shifted_meth[n] > 0) {
                frechet_lpdf_meth[n] = frechet_lpdf(y_shifted_meth[n] | a, s_norm);
            }
        }
        return frechet_lpdf_meth;
    }

    real hetero_integrand(real x, real xc, array[] real theta_array, array[] real x_r, array[] int x_i) {
        real nu = theta_array[1];
        real zeta = theta_array[2];
        real tau = theta_array[3];
        real theta = theta_array[4]; // theta renamed for readability
        real z = x_r[1];
        int T = x_i[1];

       
        vector[3] frechet_parameters_nu = calculate_frechet_parameters(nu, theta, T - tau);
        real a = frechet_parameters_nu[1];
        real m1_norm = frechet_parameters_nu[2];
        real s1_norm = frechet_parameters_nu[3];   
        vector[3] frechet_parameters_zeta = calculate_frechet_parameters(zeta, theta, T - tau);
        real m2_norm = frechet_parameters_zeta[2];
        real s2_norm = frechet_parameters_zeta[3]; 

        real shifted1 = x - m2_norm;
        real shifted2 = z - 0.5 + x - m1_norm;

        if (shifted1 > 0 && shifted2 > 0 ) {
            return exp(frechet_lpdf(shifted1 | a, s2_norm) + frechet_lpdf(shifted2 | a, s1_norm));
        } else {
            return 0.0; 
        }

    }

    vector hetero_meth_logpdf(data vector y, real theta, real nu, real zeta, real tau, data int T, int N) {
        vector[N] vector_logpdf;
        vector[3] frechet_param_nu = calculate_frechet_parameters(nu, theta, T-tau);
        vector[3] frechet_param_zeta = calculate_frechet_parameters(zeta, theta, T-tau);
        real s1_norm = frechet_param_nu[2];
        real s2_norm = frechet_param_zeta[2];
        for (n in 1:N) {
            vector_logpdf[n] = log(integrate_1d(hetero_integrand, fmax(0.0, fmax(s2_norm, s1_norm - y[n] + 0.5)), 1.0, {nu, zeta, tau, theta}, {y[n]}, {T}));   
        }
        return vector_logpdf;
    }

    vector combined_logpdf(data vector y, real theta, real tau, real mu, real gamma, real nu, real zeta, data int T, int N) {
        vector[3] mixing_logpdf = calculate_mixing_weights(mu, gamma, nu, zeta, tau);
        vector[N] left_peak = homo_demeth_logpdf(y, theta, gamma, tau, T, N);
        vector[N] central_peak = hetero_meth_logpdf(y, theta, nu, zeta, tau, T, N);
        vector[N] right_peak = homo_meth_logpdf(y, theta, mu, tau, T, N);

        vector[N] weighted_left_peak = mixing_logpdf[1] + left_peak;
        vector[N] weighted_central_peak = mixing_logpdf[2] + central_peak;
        vector[N] weighted_right_peak = mixing_logpdf[3] + right_peak;

        vector[N] combined_logpdf;
        for (n in 1:N) {
            combined_logpdf[n] = log_sum_exp([weighted_left_peak[n], weighted_central_peak[n], weighted_right_peak[n]]');
        }
        return combined_logpdf;
    }
}

data {
    int<lower=0> N; // number of fCpG loci to simulate  
    vector<lower=0, upper=1>[N] y; // measured population at measurement times
    int<lower=1> T; // patient's age
}

transformed data{
    real SminLog = log(10^2);  // log(minimum effective population size)
    real SmaxLog = log(10^9);  // log(maximum effective population size)
}

parameters{
    real<lower=0> mu; // homo_demeth to hetero transition rate
    // real<lower=0> gamma; // homo_meth to hetero transition rate
    // real<lower=0> nu_rel; // Commented out as it doesn't affect homo demeth 
    // real<lower=0> zeta_rel; //  
    real<lower=0, upper=1> tau_rel; // Relative Time until population begins growing exponentially
    real<lower=0> theta; // exponential growth rate of population
}

transformed parameters {
    real<lower=0> tau = tau_rel*T;
    // real<lower=0> nu = nu_rel * mu;
    //  real<lower=0> zeta = zeta_rel * gamma;
    real<lower=0> pop_size_log = theta * T * (1 - tau_rel);       // effective population size since the MRCA
}

model {
    // Priors
    theta ~ lognormal(3,2);
    tau_rel ~ beta(2,2);
    mu ~ normal(0,0.05);
    // gamma ~ normal(0,0.05);
    // nu_rel ~ lognormal(1, 0.7);
    // zeta_rel ~ lognormal(1,0.7);

    // When you're testing just one of the peaks, you can comment out all 
    // the paramater terms + priors that aren't relevant (e.g. if you're doing 
    // homo_demeth_lpdf(theta, mu, tau, T) as below, you can comment out the gamma,
    // nu and zeta definitions and priors)

    // Penalise the likelihood to ensure that the effective population size
    // is constrained within reaosnable bounds (~95% between Smin and Smax)
    pop_size_log ~ normal((SminLog + SmaxLog) / 2, (SmaxLog + SminLog) / 4);

    // Model
    target += sum(homo_demeth_logpdf(y, theta, mu, tau, T, N));
}

generated quantities {
    //vector[3] log_prob_states;
    //log_prob_states = calculate_mixing_weights(mu, gamma, nu, zeta, tau);

    vector[3] frechet_parameters_demeth;
    frechet_parameters_demeth = calculate_frechet_parameters(2*mu, theta, T - tau);


    real log_lik_demeth;
    log_lik_demeth = sum(homo_demeth_logpdf(y, theta, mu, tau, T, N));

    // vector[N] m; // the below code is used for graphing the peaks and the frechet distributions that approximate them
    // vector[N] logpdf_demeth;
    // vector[N] logpdf_meth;
    // vector[N] logpdf_hetero;

    // for (n in 1:N) {
    //     m[n] = fmin((n-1)/1001.0, 1);
    // }
    // logpdf_demeth = homo_demeth_logpdf(m, 2.4, 0.01, 45, 50, 10000);
    // logpdf_meth = homo_meth_logpdf(m, 2.4, 0.01, 45, 50, 10000);
    // logpdf_hetero = hetero_meth_logpdf(m, 2.4, 0.01, 0.01, 45, T, 10000);
}