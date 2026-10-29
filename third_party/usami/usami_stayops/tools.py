from usami_harness.tools import schemas as _schemas,parse as _parse

DESCRIPTIONS={
    'create_hold':'Create a temporary room-type booking at the resource property. Give exact dates, guest, plan, occupants, new reservation ID and hold_minutes (15/30/60). Every night must have capacity.',
    'capture_payment':'Simulate an integer-cent payment on the reservation. card_approved succeeds and card_declined fails. Same key/value retry is idempotent.',
    'confirm_reservation':'Confirm a valid unexpired hold after its quoted deposit is successfully paid.',
    'release_hold':'Release a temporary hold without a cancellation fee, explicitly freeing all date inventory.',
    'expire_hold':'Explicitly expire a hold only when the frozen clock reaches its expiry, freeing all date inventory. No tool advances time.',
    'change_dates':'Revalidate the entire new interval, release old room assignment and return the booking to a repriced hold requiring reconfirmation.',
    'change_room_type':'Upgrade/downgrade an unoccupied booking to a permitted type/plan, revalidate all nights, release the room and return to a repriced hold.',
    'assign_room':'Assign a clean, nonmaintenance room of the confirmed booking type without overlapping another active assignment.',
    'unassign_room':'Release a confirmed booking room assignment before check-in while retaining type inventory and deposit.',
    'cancel_reservation':'Cancel a confirmed booking, release date inventory and assignment, record its frozen-time rate-plan fee. Use release_hold for tentative bookings.',
    'refund_reservation':'Refund with an immutable key, bounded by actual paid balance less cancellation fees or real active-booking overpayment.',
    'mark_room_clean':'Housekeeping or manager marks an unoccupied room clean. An occupied room cannot be cleaned through this operation.',
    'set_room_maintenance':'Manager locks/unlocks a room for maintenance. Locking cannot invalidate existing assignments or room-type bookings.',
    'check_in':'Front desk/manager checks in a confirmed booking within its dates, with room assignment, full payment and independent guest identity verification.',
    'swap_room':'Atomically move a checked-in stay to another clean available room of the same type/property; release old occupancy, dirty old room, update assignment and stay.',
    'check_out':'Check out an occupied stay, release assignment/date inventory, clear current room stay and mark the room dirty.',
    'change_calendar_limit':'Manager changes the sale limit of a resource calendar-day ID, bounded by physical type rooms and existing bookings.',
    'set_calendar_closed':'Manager closes/opens one calendar-day ID; closing is forbidden while that night has bookings or holds.',
    'verify_guest':'Front desk asks the independent identity simulator to verify the reservation guest. Do not supply an identity outcome or external attestation.',
    'get_property':'Query public property policy and configuration.',
    'get_availability':'Query public per-night capacity, held/booked counts and available inventory for the property/type/date interval.',
    'get_room':'Query scoped staff room state; a guest assigned here sees public room identification only.',
    'get_reservation':'Query a permitted reservation, quote, payments, current assignment and observed verification status.',
    'get_guest':'Query a permitted guest profile without private simulator fixtures or other booking records.',
    'get_payment_ledger':'Finance/manager or owning guest queries payments, actual balance, deposit and refundable amount.',
    'get_calendar_day':'Staff query one scoped date inventory control.',
    'get_rate_plan':'Query public deposit and cancellation policy for a property rate plan.',
    'get_customer_context':'Discover public property catalog and scoped room, guest, reservation and financial records for the fixed identity.',
}


def schemas(names):
    from . import adapter
    return _schemas(adapter,names)


def parse(name,arguments,names):
    from . import adapter
    return _parse(adapter,name,arguments,names)
