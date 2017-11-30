import optimizer_functions.front_pass as front_pass
import optimizer_functions.back_prop as back_prop


class optimizer_function:

	def __init__(self, optimizer):
		self.maxnorm = optimizer.maxnorm
		self.dropout = optimizer.dropout
		self.weight_decay = optimizer.weight_decay
		self.scaling_noise = optimizer.scaling_noise
		self.gaussian_noise = optimizer.gaussian_noise

	@property
	def back_prop(self):
		if self.dropout:
			if self.scaling_noise:
				return back_prop.scale_dropout_back_prop
			else:
				return back_prop.dropout_back_prop
		else:
			if self.scaling_noise:
				return back_prop.scale_back_prop
			else:
				return back_prop.back_prop


	@property
	def front_pass(self):
		if self.weight_decay:
			if self.dropout:
				if self.gaussian_noise:
					if self.scaling_noise:
						return front_pass.scaling_gauss_dropout_decay_pass
					else:
						return front_pass.gauss_dropout_decay_pass
				else:
					if self.scaling_noise:
						return front_pass.scaling_dropout_decay_pass
					else:
						return front_pass.dropout_decay_pass
			else:
				if self.gaussian_noise:
					if self.scaling_noise:
						return front_pass.scaling_gauss_decay_pass
					else:
						return front_pass.gauss_decay_pass
				else:
					if self.scaling_noise:
						return front_pass.scaling_decay_pass
					else:
						return front_pass.decay_pass
		else:
			if self.dropout:
				if self.gaussian_noise:
					if self.scaling_noise:
						return front_pass.scaling_gauss_dropout_front_pass
					else:
						return front_pass.gauss_dropout_front_pass
				else:
					if self.scaling_noise:
						return front_pass.scaling_dropout_front_pass
					else:
						return front_pass.dropout_front_pass
			else:
				if self.gaussian_noise:
					if self.scaling_noise:
						return front_pass.scaling_gauss_front_pass
					else:
						return front_pass.gauss_front_pass
				else:
					if self.scaling_noise:
						return front_pass.scaling_front_pass
					else:
						return front_pass.front_pass


