from usami_harness.tools import schemas as _schemas, parse as _parse

DESCRIPTIONS = {
    'receive_inventory': 'Consume a supplier receipt notice once; resource is notice ID, params identifies destination lot and idempotency key.',
    'inspect_lot': 'Ask the independent stock inspector to release a quarantined lot. No caller-provided quality outcome.',
    'dispatch_transfer': 'Move available source-lot goods into transit, specifying destination warehouse and lot. Both warehouse scopes require authority.',
    'receive_transfer': 'Consume transfer carrier proof at the destination, with receiving, category and physical capacity checks.',
    'create_order': 'Create a draft order at the resource warehouse with customer and SKU lines; new order ID must be unused.',
    'modify_order_line': 'Change a draft order line quantity; resource is order ID. Recalculate payment required before confirmation.',
    'capture_payment': 'Simulate a charge on the order. card_approved succeeds, card_declined fails. Exact same key retries are idempotent.',
    'confirm_order': 'Confirm a fully paid draft order; payment alone does not confirm it.',
    'reserve_inventory': 'Allocate available quality-released lot stock to a confirmed order line in this warehouse.',
    'release_reservation': 'Release an unpicked allocation after voiding any shipping label linked to it.',
    'pick_reservation': 'Move reserved shelf stock to picked staging with a physical pick record.',
    'unpick_reservation': 'Reverse picked staging to reserved stock after unpacking and voiding linked labels.',
    'pack_reservation': 'Pack a picked allocation and create its packing record.',
    'unpack_reservation': 'Reverse packing before dispatch; void any linked shipping label first.',
    'create_shipment': 'Create an idempotent shipping label for selected allocations of an order. Labeling may precede packing; dispatch cannot.',
    'cancel_shipment': 'Void a label before dispatch, releasing its allocation links without changing physical goods.',
    'dispatch_shipment': 'Dispatch only fully picked and packed allocations; host carrier simulator issues delivery evidence.',
    'confirm_delivery': 'Consume independent carrier evidence for a dispatched shipment; do not supply an outcome.',
    'cancel_order': 'Cancel an entirely unshipped order, releasing shelf allocations and labels. Physically reverse picks/packs first.',
    'cancel_order_line': 'Cancel an unshipped quantity of an order line, using free quantity then releasable shelf allocations.',
    'request_return': 'Request a partial or full return against a delivered shipment allocation within product policy and delivered quantity.',
    'cancel_return': 'Cancel a return request before physical receipt.',
    'receive_return': 'Receive independently attested returned goods into quarantine if warehouse receiving and capacity permit.',
    'inspect_return': 'Ask an independent inspector to determine resalable, scrapped and refundable units. Customer reason does not decide grade.',
    'restock_return': 'Move accepted inspected goods back to their original lot once; damaged goods cannot be restocked.',
    'refund_order': 'Refund the order with an idempotency key, bounded by successful payments and canceled or independently inspected eligible goods.',
    'change_warehouse_capacity': 'Manager changes integer physical volume capacity; it cannot be below current physical usage.',
    'set_warehouse_receiving': 'Manager enables or disables inbound receiving at this warehouse.',
    'get_warehouse': 'Query scoped warehouse configuration and current physical volume usage.',
    'get_inventory': 'Staff query lot buckets: shelf on-hand, reserved, available, picked and gross shipped quantities.',
    'get_catalog': 'Query public product catalog and pricing for a warehouse organization; private inspection fixtures are excluded.',
    'get_order': 'Query a permitted order, line quantities, price due and actual fulfillment totals.',
    'get_shipment': 'Query a permitted shipment and delivery status, excluding simulator internals.',
    'get_returns': 'Query return requests and inspection results for a permitted order.',
    'get_payment_ledger': 'Finance/manager or owning customer query order charges, refunds and remaining refundable amount.',
    'get_customer_context': 'Discover permitted warehouse catalog, orders, stock, roles and receipt notices for the fixed authenticated identity.',
}


def schemas(names):
    from . import adapter
    return _schemas(adapter, names)


def parse(name, arguments, names):
    from . import adapter
    return _parse(adapter, name, arguments, names)
