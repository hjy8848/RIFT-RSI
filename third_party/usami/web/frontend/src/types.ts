export type Role = 'owner' | 'billing_admin' | 'member';
export type Mode = 'reference' | 'sandbox';
export type Page = 'overview' | 'organizations' | 'graph' | 'playground' | 'scenarios' | 'timeline' | 'benchmark';
export type Json = unknown;
export interface User { id: string; name: string; email: string }
export interface Membership { id: string; org_id: string; user_id: string; role: Role; status: string }
export interface Subscription { id: string; org_id: string; plan_id: string; seats: number; auto_renew: boolean; status: string; period_start: string; period_end: string; service_invoice_id?: string | null }
export interface Payment { id: string; key: string; amount_cents: number; method: string; outcome: string; attempted_at: string }
export interface Invoice { id: string; org_id: string; amount_cents: number; refundable_amount_cents: number; status: string; paid_at: string | null; payment_method: string | null; subscription_id?: string; period_start?: string; period_end?: string; purpose?: string; payments?: Payment[]; refunds: { key?: string; id: string; amount_cents: number; payment_method: string; refunded_at: string }[] }
export interface Plan { id: string; name: string; max_seats: number; rank: number }
export interface Organization { id: string; name: string; owner_id: string; subscription_id: string }
export interface World { world_id: string; seed: number; clock: string; generator_version: string; config_hash: string; rules: { version: string; refund_window_days: number; plans: Record<string, Plan> }; users: Record<string, User>; organizations: Record<string, Organization>; memberships: Record<string, Membership>; subscriptions: Record<string, Subscription>; invoices: Record<string, Invoice> }
export interface OrganizationView extends Organization { owner: User; subscription: Subscription; plan: Plan | null; active_members: number; memberships: Membership[]; invoice_count: number }
export interface Violation { code: string; entity_id: string; event_id: string | null; state_hash: string | null; state_path: string | null; detail: string }
export interface Evaluation { world_id: string; request_id: string; valid_request: boolean | null; operation_correct: boolean | null; violations: Violation[]; evidence_event_ids: string[] }
export interface Report { status: string; trace_valid: boolean; requests: Evaluation[]; violations: Violation[]; metrics: Record<string, number>; final_state_hash: string }
export interface Operation { name: string; org_id: string; resource_id: string; params: Record<string, string | number> }
export interface RequestContext { request_id: string; requester_id: string; org_id: string; resource_id: string; kind: string; operation: Operation; authorized_operations: Operation[] }
export interface EventSummary { event_id: string; kind: string; payload: { request_id?: string; actor_id?: string; operation?: Operation; result?: { success: boolean; error_code: string | null; data: Record<string, Json> }; context?: RequestContext; report?: Report; [key: string]: Json }; before_hash: string | null; after_hash: string | null }
export interface FullEvent extends Omit<EventSummary, 'before_hash' | 'after_hash'> { before: World | null; after: World | null; event_hash: string; previous_hash: string }
export interface Config { user_count: number; organization_count: number; profile: string; clock: string; version: string; workflow?: string | null; rules: World['rules'] }
export interface Snapshot { exploration_source?: { suite: string; instance_id: string; customer_identity: string; kind: string } | null; world: World; config: Config; initial_state_hash: string; state_hash: string; trace_initial_state_hash: string; summary: { organizations: number; users: number; active_subscriptions: number; invoices: number }; organizations: OrganizationView[]; evaluation: Report; events: EventSummary[]; operations_used: number; operation_budget: number; invariants: { code: string; entity_id: string; state_path: string; detail: string }[] }
export interface OperationView { event: FullEvent; changes: { path: string; before: Json; after: Json }[]; evaluation: Evaluation; reference_allowed: boolean | null; reference_error_code: string | null; reason: string }
export interface ExecutionResponse { snapshot: Snapshot; operation: OperationView }
export interface Scenario { scenario_id: string; category: string; initial_state_hash: string; operation_budget: number; steps: { request: RequestContext; expected_allowed: boolean; expected_error_code: string | null; expected_state_hash: string }[] }
export interface ScenarioResponse { snapshot: Snapshot; steps: OperationView[]; scenario_id: string }
export interface GraphNode { id: string; kind: string; label: string; entity: Record<string, Json>; memberships?: Membership[] }
export interface GraphEdge { id: string; source: string; target: string; label: string; kind: string; entity?: Membership }
export interface GraphData { nodes: GraphNode[]; edges: GraphEdge[]; state_hash: string }
