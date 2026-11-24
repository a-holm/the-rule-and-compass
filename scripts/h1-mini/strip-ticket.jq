del(.timing, .elapsed_seconds, .gpu, .dispatch.route_ledger, .checkpoint.resumed_from, .checkpoint.saved_to)
| walk(if type == "object" then with_entries(select(.key | test("(_s|_seconds|_per_s)$") | not)) else . end)
