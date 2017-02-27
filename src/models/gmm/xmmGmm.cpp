/*
 * xmmGmm.hpp
 *
 * Gaussian Mixture Model for Continuous Recognition and Regression
 * (Multi-class)
 *
 * Contact:
 * - Jules Francoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Francoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frederic
 * Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
 * Movement Interaction team <http://ismm.ircam.fr> of the
 * STMS Lab - IRCAM, CNRS, UPMC (2011-2015).
 *
 * Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.
 *
 * This File is part of XMM.
 *
 * XMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with XMM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "xmmGmm.hpp"
#include "../kmeans/xmmKMeans.hpp"

xmm::GMM::GMM(bool bimodal) : Model<SingleClassGMM, GMM>(bimodal) {}

xmm::GMM::GMM(GMM const& src)
    : Model<SingleClassGMM, GMM>(src), results(src.results) {}

xmm::GMM::GMM(Json::Value const& root) : Model<SingleClassGMM, GMM>(root) {}

xmm::GMM& xmm::GMM::operator=(GMM const& src) {
    if (this != &src) {
        Model<SingleClassGMM, GMM>::operator=(src);
        results = src.results;
    }
    return *this;
}

void xmm::GMM::updateResults() {
    double maxLogLikelihood = 0.0;
    double normconst_instant(0.0);
    double normconst_smoothed(0.0);
    int i(0);
    for (auto it = this->models.begin(); it != this->models.end(); ++it, ++i) {
        results.instant_likelihoods[i] = it->second.results.instant_likelihood;
        results.smoothed_log_likelihoods[i] = it->second.results.log_likelihood;
        results.smoothed_likelihoods[i] =
            exp(results.smoothed_log_likelihoods[i]);

        results.instant_normalized_likelihoods[i] =
            results.instant_likelihoods[i];
        results.smoothed_normalized_likelihoods[i] =
            results.smoothed_likelihoods[i];

        normconst_instant += results.instant_normalized_likelihoods[i];
        normconst_smoothed += results.smoothed_normalized_likelihoods[i];

        if (i == 0 || results.smoothed_log_likelihoods[i] > maxLogLikelihood) {
            maxLogLikelihood = results.smoothed_log_likelihoods[i];
            results.likeliest = it->first;
        }
    }

    i = 0;
    for (auto it = this->models.begin(); it != this->models.end(); ++it, ++i) {
        results.instant_normalized_likelihoods[i] /= normconst_instant;
        results.smoothed_normalized_likelihoods[i] /= normconst_smoothed;
    }
}

#pragma mark -
#pragma mark Performance
void xmm::GMM::reset() {
    results.instant_likelihoods.resize(size());
    results.instant_normalized_likelihoods.resize(size());
    results.smoothed_likelihoods.resize(size());
    results.smoothed_normalized_likelihoods.resize(size());
    results.smoothed_log_likelihoods.resize(size());
    if (shared_parameters->bimodal.get()) {
        unsigned int dimension_output = shared_parameters->dimension.get() -
                                       shared_parameters->dimension_input.get();
        results.output_values.resize(dimension_output);
        results.output_covariance.assign(
            (configuration.covariance_mode.get() ==
             GaussianDistribution::CovarianceMode::Full)
                ? dimension_output * dimension_output
                : dimension_output,
            0.0);
    }
    for (auto& model : models) {
        model.second.reset();
    }
}

void xmm::GMM::filter(std::vector<float> const& observation) {
    checkTraining();
    int i(0);
    for (auto& model : models) {
        results.instant_likelihoods[i] = model.second.filter(observation);
        i++;
    }

    updateResults();

    if (shared_parameters->bimodal.get()) {
        unsigned int dimension = shared_parameters->dimension.get();
        unsigned int dimension_input = shared_parameters->dimension_input.get();
        unsigned int dimension_output = dimension - dimension_input;

        if (configuration.multiClass_regression_estimator ==
            MultiClassRegressionEstimator::Likeliest) {
            results.output_values =
                models[results.likeliest].results.output_values;
            results.output_covariance =
                models[results.likeliest].results.output_covariance;

        } else {
            results.output_values.assign(dimension_output, 0.0);
            results.output_covariance.assign(
                (configuration.covariance_mode.get() ==
                 GaussianDistribution::CovarianceMode::Full)
                    ? dimension_output * dimension_output
                    : dimension_output,
                0.0);

            int i(0);
            for (auto& model : models) {
                for (int d = 0; d < dimension_output; d++) {
                    results.output_values[d] +=
                        results.smoothed_normalized_likelihoods[i] *
                        model.second.results.output_values[d];
                    if ((configuration.covariance_mode.get() ==
                         GaussianDistribution::CovarianceMode::Full)) {
                        for (int d2 = 0; d2 < dimension_output; d2++)
                            results
                                .output_covariance[d * dimension_output + d2] +=
                                results.smoothed_normalized_likelihoods[i] *
                                model.second.results
                                    .output_covariance[d * dimension_output +
                                                       d2];
                    } else {
                        results.output_covariance[d] +=
                            results.smoothed_normalized_likelihoods[i] *
                            model.second.results.output_covariance[d];
                    }
                }
                i++;
            }
        }
    }
}

#pragma mark > Conversion & Extraction
xmm::GMM xmm::GMM::getBimodal(unsigned int dimension_input)
{
    checkTraining();
    if (shared_parameters->bimodal.get())
        throw std::runtime_error("The model is already bimodal");
    if (shared_parameters->dimension_input.get() >= shared_parameters->dimension.get())
        throw std::out_of_range("Request input dimension exceeds the current dimension");

    xmm::GMM target_model(*this);
    target_model.shared_parameters->bimodal.set(true);
    target_model.shared_parameters->dimension_input.set(dimension_input);
    for (auto it=target_model.models.begin(); it != target_model.models.end(); ++it) {
        for (int c = 0; c < it->second.components.size(); c++) {
            it->second.components[c] = it->second.components[c].getBimodal(dimension_input);
        }
    }
    target_model.reset();
    return target_model;
}

xmm::GMM xmm::GMM::getUnimodal()
{
    checkTraining();
    if (!shared_parameters->bimodal.get())
        throw std::runtime_error("The model is already unimodal");
    
    xmm::GMM target_model(*this);
    target_model.shared_parameters->bimodal.set(false);
    for (auto it=target_model.models.begin(); it != target_model.models.end(); ++it) {
        for (int c = 0; c < it->second.components.size(); c++) {
            it->second.components[c] = it->second.components[c].getUnimodal();
        }
    }
    target_model.reset();
    return target_model;
}

xmm::GMM xmm::GMM::extractSubmodel(std::vector<unsigned int>& columns) const
{
    checkTraining();
    if (columns.size() > shared_parameters->dimension.get())
        throw std::out_of_range("requested number of columns exceeds the dimension of the current model");
    for (unsigned int column=0; column<columns.size(); ++column) {
        if (columns[column] >= shared_parameters->dimension.get())
            throw std::out_of_range("Some column indices exceeds the dimension of the current model");
    }
    xmm::GMM target_model(*this);
    target_model.shared_parameters->bimodal.set(false);
    target_model.shared_parameters->dimension.set(static_cast<unsigned int>(columns.size()));
    for (auto it=target_model.models.begin(); it != target_model.models.end(); ++it) {
        for (int c = 0; c < it->second.components.size(); c++) {
            it->second.components[c] = it->second.components[c].extractSubmodel(columns);
        }
    }
    target_model.reset();
    return target_model;
}

xmm::GMM xmm::GMM::extractSubmodel_input() const
{
    checkTraining();
    if (!shared_parameters->bimodal.get())
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_input(shared_parameters->dimension_input.get());
    for (unsigned int i=0; i<shared_parameters->dimension_input.get(); ++i) {
        columns_input[i] = i;
    }
    return extractSubmodel(columns_input);
}

xmm::GMM xmm::GMM::extractSubmodel_output() const
{
    checkTraining();
    if (!shared_parameters->bimodal.get())
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns_output(shared_parameters->dimension.get() - shared_parameters->dimension_input.get());
    for (unsigned int i=shared_parameters->dimension_input.get(); i<shared_parameters->dimension.get(); ++i) {
        columns_output[i-shared_parameters->dimension_input.get()] = i;
    }
    return extractSubmodel(columns_output);
}

xmm::GMM xmm::GMM::extract_inverse_model() const
{
    checkTraining();
    if (!shared_parameters->bimodal.get())
        throw std::runtime_error("The model needs to be bimodal");
    std::vector<unsigned int> columns(shared_parameters->dimension.get());
    for (unsigned int i=0; i<shared_parameters->dimension.get()-shared_parameters->dimension_input.get(); ++i) {
        columns[i] = i+shared_parameters->dimension_input.get();
    }
    for (unsigned int i=shared_parameters->dimension.get()-shared_parameters->dimension_input.get(), j=0;
         i<shared_parameters->dimension.get();
         ++i, ++j) {
        columns[i] = j;
    }
    GMM target_model = extractSubmodel(columns).getBimodal(shared_parameters->dimension.get()-shared_parameters->dimension_input.get());
    return target_model;
}
