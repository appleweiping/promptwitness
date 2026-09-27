"""Shared prune-safe cadence adaptation of the fixed DSPy MIPRO objective.

Derived from DSPy da1736e21ffda8cc4b86379d4748b011764d507c,
dspy/teleprompt/mipro_optimizer_v2.py::_optimize_prompt_parameters.
The MIT notice below covers the reproduced original method. Only this method
is adapted: its scheduled full evaluation runs before propagating a candidate
prune. Cadence counts actual objective calls, not synthetic full trials;
an empty or already-fully-evaluated pool is logged without fabricating a score.
Original proposer, bootstrap, TPE, sampling and full-selector methods remain.
"""

# MIT License
#
# Copyright (c) 2023 Stanford Future Data Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections import defaultdict
from typing import Any


def cadence_optimizer(native):
    """Bind the explicit adapted method to the current owned native module."""

    def optimize(
        self,
        program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        evaluate: Any,
        valset: list,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Any | None:
        logger = native.logger
        GREEN, ENDC = native.GREEN, native.ENDC
        optuna = native._import_optuna()

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different "
            "combinations of instructions and few-shot examples to find the optimal "
            "combination using Bayesian Optimization.\n"
        )

        # Compute the adjusted total trials that we will run (including full evals)
        run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_num_trials = int(
            (
                num_trials
                + num_trials // minibatch_full_eval_steps
                + 1
                + run_additional_full_eval_at_end
            )
            if minibatch
            else num_trials
        )
        logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

        default_score = native.eval_candidate_program(
            len(valset), valset, program, evaluate, self.rng
        ).score
        logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = native.save_candidate_program(
            program, self.log_dir, -1
        )
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deepcopy()

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}
        objective_calls = 0

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, score_data
            nonlocal objective_calls

            objective_calls += 1
            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
            else:
                logger.info(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {
                "native_trial_number": trial.number,
                "objective_call": objective_calls,
            }

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            chosen_params, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            if self.verbose:
                logger.info("Evaluating the following candidate program...\n")
                native.print_full_program(candidate_program)

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            prune_error = None
            try:
                score = native.eval_candidate_program(
                    batch_size, valset, candidate_program, evaluate, self.rng
                ).score
                total_eval_calls += batch_size

                # Update best score and program
                if not minibatch and score > best_score:
                    best_score = score
                    best_program = candidate_program.deepcopy()
                    logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

                # Log evaluation results
                score_data.append(
                    {
                        "score": score,
                        "program": candidate_program,
                        "full_eval": batch_size >= len(valset),
                    }
                )  # score, prog, full_eval
                if minibatch:
                    self._log_minibatch_eval(
                        score,
                        best_score,
                        batch_size,
                        chosen_params,
                        score_data,
                        trial,
                        adjusted_num_trials,
                        trial_logs,
                        trial_num,
                        candidate_program,
                        total_eval_calls,
                    )
                else:
                    self._log_normal_eval(
                        score,
                        best_score,
                        chosen_params,
                        score_data,
                        trial,
                        num_trials,
                        trial_logs,
                        trial_num,
                        valset,
                        batch_size,
                        candidate_program,
                        total_eval_calls,
                    )
                categorical_key = ",".join(map(str, chosen_params))
                param_score_dict[categorical_key].append(
                    (score, candidate_program, raw_chosen_params),
                )
            except optuna.TrialPruned as error:
                # Never place a partial or fabricated value in the TPE/selector.
                prune_error = error
                trial_logs[trial_num]["pruned"] = True

            # If minibatch, perform full evaluation at intervals (and at the very end)
            if minibatch and (
                (objective_calls % minibatch_full_eval_steps == 0)
                or (objective_calls == num_trials)
            ):
                if not any(key not in fully_evaled_param_combos for key in param_score_dict):
                    # No unseen actual minibatch winner remains to promote.
                    # Neither an empty nor an already-full pool is a zero trial.
                    trial_logs[trial_num]["scheduled_full_evaluation"] = (
                        "no_actual_candidate"
                        if not param_score_dict
                        else "no_unseen_actual_candidate"
                    )
                else:
                    trial_logs[trial_num]["scheduled_full_evaluation"] = "actual_survivor"
                    best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                        trial_num,
                        adjusted_num_trials,
                        param_score_dict,
                        fully_evaled_param_combos,
                        evaluate,
                        valset,
                        trial_logs,
                        total_eval_calls,
                        score_data,
                        best_score,
                        best_program,
                        study,
                        instruction_candidates,
                        demo_candidates,
                    )

            if prune_error is not None:
                raise prune_error
            return score

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update(
                {f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))}
            )

        # Add default run as a baseline in optuna.
        # Native TODO: weight this by the number of evaluated samples.
        trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(
                program, instruction_candidates, demo_candidates
            ),
            value=default_score,
        )
        study.add_trial(trial)
        study.optimize(objective, n_trials=num_trials)

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls
            sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
            # Attach all minibatch programs
            best_program.mb_candidate_programs = [
                score_data
                for score_data in sorted_candidate_programs
                if not score_data["full_eval"]
            ]
            # Attach full-trainset programs in descending score order.
            best_program.candidate_programs = [
                score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
            ]

        logger.info(f"Returning best identified program with score {best_score}!")

        return best_program

    return optimize
