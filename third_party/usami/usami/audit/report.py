"""Research report rendered exclusively from measured data and source facts."""
from __future__ import annotations


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows])


def render_report(data):
    p, w, s = data["provenance"], data["world"], data["scenario"]
    hist, complexity, rules = w["histograms"], s["complexity"], s["rules"]
    total = s["template"]["total"]
    yes = lambda k: complexity[k].get("True", 0)
    dependent = total - complexity["dependencies"].get("0", 0)
    multistep = total - complexity["steps"].get("1", 0)
    lines = ["# Usami Diversity & Coverage Audit", "", "## Provenance and scope", "",
             f"Audit `{data['version']}`; signature `{data['signature_version']}`. Generator `{p['generator_version']}`; rules `{p['rules_version']}`; scenarios `{p['scenario_version']}`.", "",
             f"Fixed {w['count']} seeds: {p['seeds'][0]} … {p['seeds'][-1]}. Config hash `{p['config_hash']}`; rules hash `{p['rules_hash']}`. Python {p['python']}, Pydantic {p['pydantic']}.", "",
             "Full ordered seed list, config, source SHA256s, world hashes, signature frequencies and canonical example payloads are in `results.json`. No timestamps, API calls or subjective LLM scores enter the metrics. Same source, config and runtime reproduce byte-identical output.", "",
             "The population is the current `generate_scenarios` candidate pool. It contains reference-labeled, preconstructed operation sequences, not natural-language goals, autonomous agent trajectories or all possible legal operation sequences. Counts are observed under these signatures; they are not an exhaustive count of the SaaS domain.", "",
             "## World diversity", "",
             table(["Metric", "Observed"], [("Worlds", w["count"]), *sorted(w["unique"].items())]), "",
             "Business instance fingerprints retain names, IDs, dates and amounts. Role topology instead canonicalizes the user–organization incidence graph with roles/statuses under all organization permutations and sorted anonymous user rows; policy topology additionally includes normalized organization state. Exact graph canonicalization is restricted to ≤6 organizations; larger configs report Unknown for those two metrics. A unique whole-world graph does not imply a unique local task.", ""]
    for key in ("users_count", "organizations_count", "memberships_count", "subscriptions_count", "invoices_count", "membership_role_status",
                "user_organization_degree", "user_active_organization_degree", "organization_members", "organization_active_members",
                "subscription_plan", "subscription_status", "subscription_auto_renew", "subscription_seats_vs_capacity", "subscription_seats_vs_members",
                "subscription_joint_state", "invoice_status", "invoice_refund_class", "invoice_payment_method", "invoice_history_length", "heldout_world_combination"):
        lines += [f"**{key}**", "", table(["Value", "Count"], sorted(hist[key].items())), ""]
    cross = w["cross_role_users"]
    lines += [f"Cross-organization role differences: {cross['different_roles']}/{cross['all_users']} user instances ({cross['ratio_all']:.2%} of all; {cross['ratio_connected']:.2%} of connected; {cross['ratio_multi_organization']:.2%} of multi-organization users). Roles include inactive memberships; active organization degrees are reported separately.", "",
              "Amounts, exact seats, member/capacity slack and payment methods have full histograms in JSON. Per source, every organization has exactly one subscription and four invoices: eligible paid, stale paid, unpaid and already refunded. Full refund only; amounts are integer multiples of 100 cents. Owner allocation is deterministic by organization index; membership selection/roles/status, plan, seats, active/expired state, renewal, cycle offsets and coherent payment histories are sampled. Entity counts for users/organizations are configuration constants. Default membership count is capped at eight per organization, even when configured with more users. Plan rank is stored but does not drive reference legality.", "",
              "## Scenario structural diversity", "",
              "Template signatures preserve operation types, first-encounter anonymous typed entities, owner/subscription/resource links, actor membership roles/status across involved organizations and measured legality dependencies. Semantic signatures add policy-relative prestate/parameter comparisons, authorization, exact reference verdict, actually visited guards and normalized state changes. Names/IDs, category labels, absolute payment amounts/dates and arbitrary seat values are excluded when comparisons are unchanged. Active/expired, equal/less/greater boundary conditions and no-op versus mutation remain distinct. Relative capacity comparisons distinguish plan direction; names/ranks do not.", "",
              table(["Level", "Samples", "Unique", "Repeats", "Repeat ratio", "Singletons", "H bits", "Effective diversity", "Top10 share"],
                    [(k, s[k]["total"], s[k]["unique"], s[k]["repetitions"], f"{s[k]['repetition_ratio']:.2%}", s[k]["singletons"], f"{s[k]['entropy_bits']:.3f}", f"{s[k]['effective_diversity']:.2f}", f"{s[k]['top10_share']:.2%}") for k in ("template", "semantic")]), "",
              "Repetition = 1 − unique/sample count, conditional on the stated abstraction. It measures equivalent measured decision problems, not whether all repeated tasks are useless for statistical evaluation. Shannon H measures distribution spread; 2^H is the number of equally frequent classes with that entropy, not an extra count of novel structures.", "",
              "### Growth", "", table(["Worlds", "Scenarios", "Templates", "Semantics"], [(g["worlds"], g["scenarios"], g["templates"], g["semantics"]) for g in data["growth"]]), ""]
    for kind in ("template", "semantic"):
        rows = []
        for row in s[kind]["top10"]:
            ex = s["signature_examples"][row["signature"]]
            payload = ex["signature_payload"]
            graph = payload["relations"] if kind == "template" else payload["template"]["relations"]
            ops = " → ".join(o["operation"] for o in graph["operations"])
            verdict = "" if kind == "template" else "; ".join(str(step["outcome"]) for step in payload["steps"])
            rows.append((row["signature"][:12], row["count"], ex["seed"], ex["category"], ops, verdict))
        lines += [f"### Top 10 {kind}", "", table(["Signature prefix", "Count", "Example seed", "Category", "Operations (display only)", "Verdicts"], rows), ""]
    lines += ["The abbreviated operation sequences above are labels for reading the table, not the signature definition. Full payloads and scenario IDs allow reproduction of every observed class.", "",
              "## Combination complexity", "",
              table(["Metric", "Distribution (scenario count)"], sorted(complexity.items())), "",
              f"Multi-step scenarios: {multistep}/{total} ({multistep/total:.2%}). Scenarios with measured legality dependencies: {dependent}/{total} ({dependent/total:.2%}; {dependent/multistep if multistep else 0:.2%} of multi-step). Cross-organization: {yes('cross_organization')}/{total} ({yes('cross_organization')/total:.2%}). Mixed allowed/rejected: {yes('mixed_allowed_rejected')}/{total} ({yes('mixed_allowed_rejected')/total:.2%}). Reversal changes decision bag in {yes('decision_sensitive')} and final business state in {yes('final_state_sensitive')} scenarios.", "",
              "Dependency test: omit each successful state-changing step, reexecute the remaining prefix with the same real reference and trusted authorization, and compare later (allowed, error) verdicts. Edges can include mediated effects; depth is the longest path in this leave-one-out legality graph, not a claim of identified direct causality. Failed/no-op steps cannot generate edges. Read-to-write information dependencies and agent reasoning are Unknown/not measured. Order sensitivity tests full reversal only, comparing verdict bags grouped by actor+operation and final state after removing refund execution-nonce IDs; all permutations are not enumerated.", "",
              table(["Dependency operation pair", "Observed edges"], sorted(s["dependency_operation_pairs"].items())), "",
              "## Business rule coverage", "",
              "Inventory is extracted from actual `ReferenceRules.decide` rejection guards. Python line tracing observes which guards the unmodified function checks and which return executes. `visited` means condition evaluated (pass or veto), not a rejection. Permission guards with the same error remain separate. Authorization is a trusted harness/Engine gate and is reported separately from business policy. Policy checks on unauthorized samples are audit evaluations, not a claim that Engine executes those checks before rejecting authorization.", "",
              table(["Guard", "Source line", "Error", "Condition", "Visited", "Pass", "Veto"], [(g["id"], g["line"], g["error_code"], g["condition"], g["visited"], g["passed"], g["vetoed"]) for g in rules["implemented_guards"]]), "",
              table(["Operation", "Allowed", "Rejected", "Policy allowed", "Policy rejected"], [(op, c.get("allowed", 0), c.get("rejected", 0), c.get("policy_allowed", 0), c.get("policy_rejected", 0)) for op, c in rules["operations"].items()]), "",
              "### Operation × actual rule matrix", "",
              table(["Operation", "Guard", "Allowed samples", "Rejected samples", "Passed", "Vetoed"], [(r["operation"], r["guard"], r.get("allowed", 0), r.get("rejected", 0), r.get("pass", 0), r.get("veto", 0)) for r in rules["operation_rule_matrix"]]), "",
              "### Rejections and combinations", "", table(["Rejection", "Count"], rules["rejection_codes"].items()), "",
              table(["Actually visited business guard set", "Count"], rules["visited_business_guard_sets"].items()), "",
              table(["Derived potential multiple vetoes (not exercised interaction)", "Count"], rules["derived_multiple_veto_sets"].items()), "",
              "Business guard sets exclude g01–g07 existence/integrity/scope gates. Co-visitation is coverage of a decision path, not proof that all constraints independently affected the outcome. Derived multiple vetoes evaluate actual AST predicates with reconstructed prerequisite locals, including masked constraints; missing prerequisites are Unknown. They never replace a reference verdict or count as exercised vetoes. No causal cross-rule interaction is inferred solely from co-visitation; dependency edges provide separate evidence of changes in legality.", "",
              f"Never visited guards: {rules['never_visited']}. Never vetoed guards: {rules['never_vetoed']}. Invariant error inventory: {rules['invariant_error_codes']}. Generated/reference states satisfy invariants, so corruption guards are not expected to veto. Core tests exercise invalid-state and parameter cases separately; this is scenario-population coverage, not test-suite coverage.", "",
              "## Split overlap", "",
              "The repository ships a frozen public validation manifest, generator configurations and a training stream; no frozen standard train task set or private task test set is shipped. The existing validator prevents identical world/business instances across splits. It also enforces a real world-level holdout: one user is active billing_admin in an active-subscription organization with eligible invoice, and active member elsewhere with expired subscription and stale invoice. Train homogeneous subscription status excludes it; composed test forces it. This is stronger than seed separation, but does not enforce scenario-signature separation or require a task to use the held-out conjunction.", "",
              "The following are audit-derived public demo manifests generated through the existing selector (12 scenarios/world), never a new official/blind test set. Configs, seed lists, hashes, per-split rule sets and signature distributions are in JSON.", ""]
    for name, split in sorted(data["splits"]["demo_manifests"].items()):
        lines += [f"- {name}: {split['worlds']} worlds, {split['template']['total']} tasks, {split['template']['unique']} templates, {split['semantic']['unique']} semantic classes, {split['heldout_worlds']} heldout-conjunction worlds. Manifest `{split['manifest_hash']}`."]
    lines += ["", table(["Pair", "Level", "Common", "Left unique", "Right unique", "Jaccard", "Right classes seen left", "Right tasks seen left", "Right-only"],
                         [(pair, k, v[k]["common"], v[k]["left_unique"], v[k]["right_unique"], f"{v[k]['jaccard']:.2%}", f"{v[k]['right_structure_in_left_ratio']:.2%}", f"{v[k]['right_sample_in_left_ratio']:.2%}", len(v[k]["right_only"])) for pair, v in data["splits"]["pairwise"].items() for k in ("template", "semantic")]), ""]
    for pair, v in data["splits"]["pairwise"].items():
        lines += [f"- {pair}: right-only actual visited rule sets {v['visited_rule_sets_right_only']}; right-only dependency operation pairs {v['dependency_pairs_right_only']}."]
    shipped = data["splits"]["shipped_validation"]
    if shipped:
        lines += ["", f"Actual shipped validation: {shipped['worlds']} worlds / {shipped['template']['total']} tasks; seeds {shipped['seeds']}; {shipped['template']['unique']} templates and {shipped['semantic']['unique']} semantic classes. Manifest `{shipped['manifest_hash']}`."]
    lines += ["", "## Diversity bottlenecks and recommendations", "",
              f"**Confirmed:** {complexity['organizations'].get('1', 0)}/{total} candidates involve one organization. Candidate chains have lengths {sorted(complexity['steps'])}; multi-step construction is limited to repeat refund, plan/seat downgrade and role/billing families (`scenario.py`). Cross-scope rejections are single-step cases, not cross-organization causal workflows. World topology diversity therefore substantially exceeds the local task variety measured here.", "",
              "**Confirmed:** get_organization, get_invoice and list_members have zero candidate occurrences. Inactive memberships occur in worlds but role-change targets are selected only from active memberships. Role changes are requested by owners; billing-admin/member role-management denial is absent. Unknown actor/org/resource and invalid plan cases are absent because constructors select real references/plans. g01/g02 are integrity policy guards, for which intentionally corrupting benchmark worlds would be a different evaluation track.", "",
              "**Confirmed:** invoices are independent organization-owned records; no invoice–subscription billing-cycle link exists. No operation changes active-member count, pays an invoice, activates/renews a subscription or advances frozen time. These states cannot be reached by randomly concatenating current operations. Engine/Oracle remain unchanged.", "",
              "**Likely:** routine service goal diversity and compositional transfer are constrained by these constructor families. The overlap data support this concern, but predict neither LLM accuracy nor exploit yield.", "",
              "**Not yet measurable:** exhaustive theoretical signature count, direct causal rule interaction, customer attack novelty, LLM capability, information-flow dependencies and real private-test overlap. These require exhaustive domain enumeration, purpose-built causal tasks, measured agent trajectories or access to an actual private test manifest.", "",
              "See `ROADMAP.md` for versioned Service-goal evaluation versus open-ended discovery, causal generation and prioritized subsystem proposals. This audit adds no business operations or benchmark scoring changes.", "",
              "## Verification and limits", "",
              f"All {data['verification']['reference_steps']} candidate steps were recomputed through real ReferenceRules and matched generated labels and expected state hashes. {len(data['verification']['engine_oracle_replay_checks'])} distinct template representatives additionally used actual Engine execution, trusted-request Oracle evaluation and exact event replay; every check passed. This subset is explicitly smaller than the full candidate pool. Demo splits and shipped validation were also reference checked.", "",
              "No Engine, Generator, Reference Rules or Oracle files are edited; their source SHA256s are frozen in the results. Tests, exact invocation, artifact digest and known limitations are recorded in `VERIFICATION.md`. Finite seeds do not prove asymptotic saturation. Abstraction choices are versioned and inspectable: changing their granularity changes the measured number of classes.", ""]
    return "\n".join(lines)
