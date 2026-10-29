# Eight-domain contracts and current implementation boundary

The common source contract is `WorldSpec → FrozenWorld → isolated runtime World`. Current generator bindings exist for the four Track A domains only. The four Track B business contracts below are implementation requirements, not claims of completed native modules.

## Shared task and evidence obligations

A task must select an existing registered world, an existing actor, permitted observable objects, an objective joint goal and a finite operation scope. The task composer must prove completion through actual native transitions, or verify a specific native policy refusal. A private witness is proof material and not a prescribed Service route.

Mixed tasks must carry independently evaluable completion/refusal subgoals and a joint frame. A failed allowed subgoal is not excused merely because a different subgoal was correctly refused. Illegal successful side effects, blocked attempts, invalid task proposals, query errors and uncertain attribution remain separate facts.

Every dialogue task requires a frozen Customer profile. Completion tasks also require an independently proved information/consent dependency: a necessary process fact must change the actual allowed action set, and conversation length is descriptive only. A refusal-only task may instead be certified against its exact frozen policy boundary, with the Reference reason, Oracle decision and Replay verified independently. Open-composition v1 keeps the original dependency requirement for all tasks; v2 makes this refusal-only distinction explicit. Neither protocol lets the Designer invent entities, permissions, facts or expected outcomes.

## Track A — existing native business contracts

| Domain | State/authority | Native constraints | New generator status |
| --- | --- | --- | --- |
| SaaS | Directory, organizations, memberships, subscriptions, invoices, payments/refunds | Owner protection, tenant isolation, active seat limits, funding cycles, refund accounting | Existing unconditioned v0.2 distribution registered; 100-world audit passed |
| CloudOps | Users, organizations, projects, role edges, instances, dependency DAG, invoices | Project-local compute role, quota, prerequisite instances, suspended project, payment/resume separation | New independent structural distribution; 100 valid audited worlds |
| Fulfillment | Organizations, warehouses/roles, SKU/lots, receipts, orders, allocations, handling, shipments, returns and ledger | Conserved stock, quality, capacity, warehouse scope, attested delivery/inspection, refund eligibility | New topology plus actual lifecycle distribution; 100 valid audited worlds |
| StayOps | Property/roles, guests, physical rooms, date capacity, rate plans, reservations, assignment, stays and payments | Half-open dates, capacity, cleanliness, maintenance, quote/deposit, identity, cancellation terms | New topology/calendar/lifecycle distribution; 100 valid audited worlds; native checkout consistency limitation retained |

Existing Service and Customer interactions remain available. Describing Track A as tool-centric does not mean these environments have no dialogue.

## Track B — required native rules to implement next

### After-sales

Different order lines may have distinct problem evidence, delivery dates, remaining refund balance and replacement inventory. Case evidence must bind to the selected order/line and independently simulated evidence result. Return, refund, replacement and reshipment must have different dependencies. Refund money must follow the original payment channel, remain bounded by actual eligible paid balance and be idempotent. Ineligibility or another person's order must cause an exact refusal. Irreversible settlement needs actual current consent; a stale resolution choice must not settle.

Required private dependencies include selecting which of multiple real orders/lines is meant, describing the issue, supplying evidence held by the customer, expressing time/option constraints and confirming the actual offer. Existing entity facts come from the independently generated world; no Service or Designer may invent verification success.

### Airline

A passenger booking refers to real legs, fare rights and original charges. Change quotes must respect seat capacity, travel dates, airport continuity and minimum connection time. Price difference and change fees are deterministic. Carrier disruption and voluntary cancellation are distinct branches. Ancillaries remain tied to the actual passenger/itinerary. Irreversible exchange/cancellation requires current explicit confirmation; ambiguous intent cannot cancel the ticket.

Missing itinerary/date/arrival-limit/budget and alternative itinerary preferences affect legal choices. New disruption or seat events must be pre-registered simulator events, not invented agent messages.

### Property

A unit has real occupants/owner/contract scope, equipment and maintenance responsibilities. Incidents have objective simulator facts, priority and required technician skills. Scheduling must fit agreed household windows, technician availability, skills and absence of existing conflicts. Structural/chargeable work may require independent owner approval. Completion requires real simulated technician evidence; a Service assertion cannot complete a job. Complaint escalation is separate from actual repair.

Customer disclosure selects location/fault facts; severity updates, accessible time windows, charge confirmation and owner approval alter the permissible action set. Another household's private information is never a valid query result.

### Education

Contracts bind paid enrollment, consumed lessons, validity, course rights and refund arithmetic. Transfers consume real class capacity and must respect teacher/time constraints. Deferral/extension rights are bounded by actual terms. Refunds must account for consumed service and prior refunds; arbitrary discounts or approvals are not allowed. Scheduling and progress adjustments need separate legal transitions rather than four renamed refund workflows.

The customer may initially describe dissatisfaction without disclosing the actual reason or intended resolution. Clarified schedule, objective progress, preferred alternatives and explicit current consent determine valid changes. Previously withdrawn or superseded choices cannot be executed later.

## Runtime boundaries still to connect

The implemented `usami_dialogue` reducer verifies private-value provenance, preferences and consent scope, but does not yet validate these four industries' quotes or execute native business actions. Native domain callbacks, Engine/Oracle traces, mixed task proofs, Customer/Service sessions and unified adapters are mandatory remaining work. No common CRUD facade is an acceptable substitute.

## Executable After-sales core

The After-sales implementation (`usami_aftersales/`) is an order-line dispute state machine, with merchant-scoped ownership, simulated evidence verification, refund approval limits, paid-value/original-channel accounting, replacement stock conservation, pickup and verified receipt. Its native operations implement open-case, evidence verification, approval request, actual quote, consent-bound settlement, return scheduling and refund release. Scoped read tools expose the actual order/case/options without private profile or simulator outcomes.

The Customer profile chooses a real buyer/order/line, immutable problem/evidence facts, deadline and acceptable resolution space before execution. A Customer declaration can itself supply exact facts; missing-information rules must not manufacture extra turns after those facts have been supplied. The production model chooses whether to clarify, compare, revise, confirm or finish. Manager/carrier events have pre-registered outcomes and independent expected/actual handlers.

The current domain task contract allows actual terminal line states and explicit duplicate-resolution refusal subgoals. Frames are supplied by the domain binding and cannot be broadened by a task proposal. Native Reference search is private certification; production execution remains autonomous. Current fixtures cover two different valid resolution paths, a legal extra-query path, a mixed completion/refusal goal, consent and preference boundaries, premature Sandbox settlement, and actual simulator receipt/approval dependencies.

This is not a claim of full After-sales coverage: historical initial case distributions, richer account choices, full cross-order/multi-entity task families and the final Reference decision matrix remain under development. The three other Dialogue-Centric domains remain unimplemented at this checkpoint.

## Explicit StayOps correction version

`stayops-structural-v2` binds `stayops-0.2.0` under `usami_benchmark/native/stayops_v2/`. A currently dirty unoccupied room can have an assignment for a future arrival; actual assignment/check-in cleanliness checks and the prohibition on a currently occupied dirty room remain. This resolves the legacy checkout-versus-future-assignment invariant conflict. Both historical native `stayops-0.1.0` and the first structural generator remain available under their original explicit registrations; old traces are not re-scored under the new invariant.


## Executable Airline core

`usami_airline/` models carrier-scoped passenger bookings, dated flight legs, airports and minimum connection times, cabin inventory, fare contracts, baggage and a signed simulated financial ledger. Rebooking releases the old seat use by changing the actual booking and consumes capacity on the selected legs. Quotes do not reserve seats: native Reference rechecks capacity, current terms, fee/arrival constraints and exact consent at commit. Cancelled tickets release seat use and retain only non-refundable charges according to the executed accounting rules.

An airline's actual pre-registered cancellation is distinct from a passenger's voluntary refund request. The first can waive a fare restriction and fee; Customer statements cannot create it. External delays alter real flight times and connection feasibility. The Service can query actual fare terms, timetable, availability, baggage limits and its authenticated passenger's remaining payment capacity without seeing private simulator identity outcomes.

The fixed private passenger profile contains actual booking/document facts, travel and budget constraints, and reasonable itinerary/service choices. A certified-task runner tells the Customer which business request is selected while withholding Reference answers. The Customer can clarify, choose, revise and consent based on actual replies; it is not a timetable script. Legal commits use the same actual Engine in the fixture protocol tests and the real-model entry point.

Standalone refusal tasks require the exact requested fare/capacity/connection boundary to be independently provable. Their frames allow passenger verification and registered operational events, while protecting ticket/financial state. Mixed refund/baggage tasks additionally reject a real duplicate request after legal completion. These examples do not exhaust the required full domain coverage.


## Executable Property core

`usami_property/` ties service authority to actual building/unit ownership and residency. Assets carry distinct fault categories, responsibility and costs. Contracts define coverage, resident approval limits, service hours and response deadlines. Public staff have category skills/on-call status and overlapping appointment options; independent pre-existing commitments consume technician time. The simulator conserves single-technician capacity and never treats a reservation as completed work.

The Customer must actually disclose their observed fault/hazard and access constraints. Electrical sparks or a hazardous observation require emergency triage; an actual inspection can change the assessment and invalidate plans. Owner review is a controlled event with an immutable outcome, not a tool argument saying “approved.” Plan requester identity prevents one resident using another party's financial terms.

Appointment booking, cancellation and closure bind actual current offers. A technician event can arrive only for a scheduled, current work instruction, at the selected appointment's end. Completed work produces an independent receipt and correctly assigned charge; unresolved/missed work supports the corresponding complaint instead. Closure additionally requires customer feedback and a fresh explicit consent. Scoped tools expose physical/unit/service facts and public technician availability, excluding other residents' contacts and hidden simulator conditions.

Current task composers cover scheduling, real repair/closure, complaint/escalation, source-valid policy refusals and an appropriate mixed duplicate-closure request. The private Reference witness is not a production conversation. Current tests include a six-turn scripted-transport execution through the actual generic runner, but do not claim that real models have solved or learned these tasks. Broader initial histories, multi-entity task families and final coverage certification remain active work.

## Education native contract (implementation checkpoint)

`usami_education/` models signed study-unit contracts and actual money/study-value
ledgers, rather than treating education as another refund ticket. Students own
school-scoped enrollments, with original and current courses kept separately.
Packages bound transfer choices; verified placement levels, finite cohort seats,
dated classes, teacher qualifications and calendars constrain new arrangements.
Opening consumption records retain their actual course price. Transferring or
buying additional units does not reset cumulative consumption for refund policy.

Refunds use the unused balance, original contract window and consumed-unit limit,
subtract the contractual fee and return money to the original channel. Extensions
have reason, total-day, grace and self-service limits; an academic simulator must
actually approve requests beyond authority. A review is bound to the requested
days and enrollment version. Changing either invalidates the old approval.
Tutoring reservations reserve study credit; actual attendance/no-show outcomes
consume it according to the contract. Teacher cancellation consumes no credit.
No model can invent an attendance record, approval, discount or new payment.

The fixed Customer privately holds their genuine problem, budget, available
hours, minimum refund/study needs and bounded course/slot/extension choices.
Current exact quote consent is required for an irreversible commitment; changing
preferences or enrollment state invalidates old offers. Domain composers certify
refund, transfer, extension, tutoring, attendance and native refusal goals via
private Reference search, actual Engine execution, independent Oracle and Replay.
Production agents receive scoped queries and the Customer's own disclosures,
not the private witness, future academic outcome or reference refusal answer.

This checkpoint does not yet establish complete historical case distributions,
cross-enrollment composite tasks, balanced release quantities or model quality.
