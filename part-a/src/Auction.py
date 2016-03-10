#!/usr/bin/python
import Bid, Slot

class Auction:
	'This class represents an auction of multiple ad slots to multiple advertisers'
	query = ""
	bids = []

	def __init__(self, term, bids1=[]):
		self.query = term

		for b in bids1:
			j=0
			print(len(self.bids))
			while j<len(self.bids) and float(b.value) <float(self.bids[j].value):
				j+=1
			self.bids.insert(j,b)

	'''
	This method accepts a Vector of slots and fills it with the results
	of a VCG auction. The competition for those slots is specified in the bids Vector.
	@param slots a Vector of Slots, which (on entry) specifies only the clickThruRates
	and (on exit) also specifies the name of the bidder who won that slot,
	the price said bidder must pay,
	and the expected profit for the bidder.
	'''

	def executeVCG(self,slots):
		# TODO: implement this method
		# print ("executeVCG: To be implemented")

		price = 0.0
		lowClickThruRate = 0.0

		# for loop as many times as number of slots, descending >> if 4 slots, it will loop: 3, 2, 1, 0
		for bid in range(len(slots)-1, -1, -1):
			# if bid no. is smaller than number of bids, continue
			if bid < len(self.bids):
				# if current bid no. + 1 is smaller than number of bids, do calculations
				if bid+1 < len(self.bids):
					# calculate price: bid amount * (current slot's clickThruRate - lowClickThruRate)
					price += self.bids[bid+1].value * (slots[bid].clickThruRate - lowClickThruRate)
				# assign current bidder's price
				slots[bid].price = price
				# assign current bidder's name
				slots[bid].bidder = self.bids[bid].name
				# calculate and assign current bidder's profit: (bid amount * clickThruRate) - price
				slots[bid].profit = (self.bids[bid].value * slots[bid].clickThruRate) - slots[bid].price
			# if bid no. is larger than number of bids, assign zero values
			elif bid > len(self.bids):
				# assign zero value to current bidder's price
				slots[bid].price = 0
				# assign zero value to current bidder's name
				slots[bid].bidder = 0
				# assign zero value to current bidder's profit
				slots[bid].profit = 0
			# assign current slot's clickThruRate to lowClickThruRate
			lowClickThruRate = slots[bid].clickThruRate